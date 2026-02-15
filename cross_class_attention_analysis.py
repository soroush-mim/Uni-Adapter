"""
Cross-Class Attention Analysis for Test-Time Adaptation

This script analyzes how attention patterns differ across classes and how
corruption affects the "distance" between classes in attention space.

Key outputs:
1. Distance matrices showing class similarities (clean vs corrupted)
2. Confusion analysis: which classes become more similar under corruption
3. t-SNE visualization with displacement arrows showing class movement
4. Per-severity analysis across all 5 severity levels

Usage:
    python cross_class_attention_analysis.py --myroot /path/to/modelnet40_c --corruption cutout
    python cross_class_attention_analysis.py --myroot /path/to/modelnet40_c --corruption jitter --samples-per-class 20
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import logging
import copy
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json

# For dimensionality reduction
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. t-SNE visualizations will be disabled.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import from project
from utils.params import parse_args
from utils.load_models import load_vlm_model
from data.data_utils import load_tta_dataset
from extract_attention import AttentionExtractor


class CrossClassAttentionAnalyzer:
    """
    Analyzes attention patterns across different classes and compares
    how corruption affects class relationships in attention space.
    """

    def __init__(self, model, device: str = 'cuda:0'):
        """
        Initialize the analyzer.

        Args:
            model: Uni3D model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.extractor = AttentionExtractor(model, device=device)

        # Will be populated during analysis
        self.class_names = None
        self.num_classes = None

    def extract_attention_vector(self, pc: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """
        Extract CLS attention vector from a point cloud sample.

        Uses the last layer attention, averaged over heads.

        Args:
            pc: Point cloud [N, 3]
            rgb: RGB values [N, 3]

        Returns:
            Attention vector [num_tokens] (typically 512)
        """
        # Convert to tensors
        pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(self.device)

        if isinstance(rgb, torch.Tensor):
            rgb = rgb.numpy()
        rgb_tensor = torch.from_numpy(rgb).float().unsqueeze(0).to(self.device)

        # Combine XYZ and RGB
        feature = torch.cat([pc_tensor, rgb_tensor], dim=-1)

        # Extract attention maps
        attention_maps = self.extractor.extract(feature)

        # Get last layer index
        layer_indices = sorted([int(k.split('_')[1]) for k in attention_maps.keys()])
        last_layer_idx = layer_indices[-1]

        # Get CLS attention from last layer, average over heads
        attn = attention_maps[f'layer_{last_layer_idx}'][0]  # [heads, seq, seq]
        cls_attn = attn.mean(dim=0)[0, 1:].numpy()  # [num_tokens]

        return cls_attn

    def extract_all_attention_vectors(
        self,
        args,
        corruption: str,
        severity: int,
        samples_per_class: Optional[int] = None
    ) -> Tuple[Dict[int, List[np.ndarray]], List[str]]:
        """
        Extract attention vectors for all samples, organized by class.

        Args:
            args: Parsed arguments
            corruption: 'clean' or corruption name
            severity: Severity level (1-5)
            samples_per_class: Max samples per class (None for all)

        Returns:
            attention_by_class: Dict mapping class_id -> list of attention vectors
            class_names: List of class names
        """
        # Load dataset
        args_copy = copy.deepcopy(args)
        args_copy.corruption = corruption
        args_copy.severity = severity
        dataset = load_tta_dataset(args_copy)

        self.class_names = dataset.class_name
        self.num_classes = len(self.class_names)

        # Organize samples by class
        samples_by_class = defaultdict(list)
        for idx in range(len(dataset)):
            _, label, _, _ = dataset[idx]
            samples_by_class[label].append(idx)

        # Extract attention vectors
        attention_by_class = defaultdict(list)

        desc = f"Extracting {corruption} (severity {severity})"
        total_samples = 0

        for class_id in range(self.num_classes):
            sample_indices = samples_by_class[class_id]

            if samples_per_class is not None:
                sample_indices = sample_indices[:samples_per_class]

            total_samples += len(sample_indices)

        with tqdm(total=total_samples, desc=desc) as pbar:
            for class_id in range(self.num_classes):
                sample_indices = samples_by_class[class_id]

                if samples_per_class is not None:
                    sample_indices = sample_indices[:samples_per_class]

                for idx in sample_indices:
                    pc, label, class_name, rgb = dataset[idx]

                    try:
                        attn_vec = self.extract_attention_vector(pc, rgb)
                        attention_by_class[class_id].append(attn_vec)
                    except Exception as e:
                        logging.warning(f"Failed to extract attention for sample {idx}: {e}")

                    pbar.update(1)

        return attention_by_class, self.class_names

    def compute_class_centroids(
        self,
        attention_by_class: Dict[int, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Compute centroid (mean attention vector) for each class.

        Args:
            attention_by_class: Dict mapping class_id -> list of attention vectors

        Returns:
            centroids: Array [num_classes, attention_dim]
        """
        num_classes = len(attention_by_class)
        attention_dim = len(list(attention_by_class.values())[0][0])

        centroids = np.zeros((num_classes, attention_dim))

        for class_id in range(num_classes):
            if len(attention_by_class[class_id]) > 0:
                vectors = np.array(attention_by_class[class_id])
                centroids[class_id] = vectors.mean(axis=0)

        return centroids

    def compute_distance_matrix(
        self,
        centroids: np.ndarray,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix between class centroids.

        Args:
            centroids: Array [num_classes, attention_dim]
            metric: 'cosine' or 'euclidean'

        Returns:
            distance_matrix: Array [num_classes, num_classes]
        """
        num_classes = centroids.shape[0]
        distance_matrix = np.zeros((num_classes, num_classes))

        for i in range(num_classes):
            for j in range(num_classes):
                if metric == 'cosine':
                    # Cosine distance = 1 - cosine_similarity
                    norm_i = np.linalg.norm(centroids[i])
                    norm_j = np.linalg.norm(centroids[j])
                    if norm_i > 0 and norm_j > 0:
                        similarity = np.dot(centroids[i], centroids[j]) / (norm_i * norm_j)
                        distance_matrix[i, j] = 1 - similarity
                    else:
                        distance_matrix[i, j] = 1.0
                elif metric == 'euclidean':
                    distance_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])

        return distance_matrix

    def analyze_confusion(
        self,
        clean_distances: np.ndarray,
        corrupted_distances: np.ndarray,
        class_names: List[str],
        top_k: int = 10
    ) -> Dict:
        """
        Analyze which classes become confused under corruption.

        Args:
            clean_distances: Distance matrix for clean samples
            corrupted_distances: Distance matrix for corrupted samples
            class_names: List of class names
            top_k: Number of top confused pairs to return

        Returns:
            analysis: Dictionary containing confusion analysis
        """
        num_classes = len(class_names)

        # Distance change: negative = classes moved closer (more confused)
        distance_change = corrupted_distances - clean_distances

        # Find top confused pairs (most negative distance change, excluding diagonal)
        confused_pairs = []
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                confused_pairs.append({
                    'class_i': class_names[i],
                    'class_j': class_names[j],
                    'class_i_idx': i,
                    'class_j_idx': j,
                    'clean_distance': clean_distances[i, j],
                    'corrupted_distance': corrupted_distances[i, j],
                    'distance_change': distance_change[i, j]
                })

        # Sort by distance change (most negative first = most confused)
        confused_pairs.sort(key=lambda x: x['distance_change'])
        top_confused = confused_pairs[:top_k]

        # For each class, find which class it moved toward
        class_movement = []
        for i in range(num_classes):
            # Get distances to all other classes
            clean_dists = clean_distances[i].copy()
            corrupted_dists = corrupted_distances[i].copy()

            # Exclude self
            clean_dists[i] = np.inf
            corrupted_dists[i] = np.inf

            clean_nearest = np.argmin(clean_dists)
            corrupted_nearest = np.argmin(corrupted_dists)

            class_movement.append({
                'class': class_names[i],
                'class_idx': i,
                'clean_nearest': class_names[clean_nearest],
                'clean_nearest_idx': clean_nearest,
                'clean_nearest_dist': clean_dists[clean_nearest],
                'corrupted_nearest': class_names[corrupted_nearest],
                'corrupted_nearest_idx': corrupted_nearest,
                'corrupted_nearest_dist': corrupted_dists[corrupted_nearest],
                'neighbor_changed': clean_nearest != corrupted_nearest
            })

        # Statistics
        neighbor_changes = sum(1 for m in class_movement if m['neighbor_changed'])

        analysis = {
            'top_confused_pairs': top_confused,
            'class_movement': class_movement,
            'neighbor_change_count': neighbor_changes,
            'neighbor_change_ratio': neighbor_changes / num_classes,
            'mean_distance_change': np.mean(distance_change[np.triu_indices(num_classes, k=1)]),
            'distance_change_matrix': distance_change
        }

        return analysis


def visualize_distance_matrices(
    clean_distances: np.ndarray,
    corrupted_distances: np.ndarray,
    class_names: List[str],
    corruption: str,
    severity: int,
    save_path: str
):
    """
    Visualize distance matrices side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    num_classes = len(class_names)

    # Clean distance matrix
    im1 = axes[0].imshow(clean_distances, cmap='viridis', aspect='auto')
    axes[0].set_title('Clean Distances', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(0, num_classes, 5))
    axes[0].set_yticks(range(0, num_classes, 5))
    axes[0].set_xticklabels(range(0, num_classes, 5), fontsize=8)
    axes[0].set_yticklabels(range(0, num_classes, 5), fontsize=8)
    axes[0].set_xlabel('Class Index')
    axes[0].set_ylabel('Class Index')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Corrupted distance matrix
    im2 = axes[1].imshow(corrupted_distances, cmap='viridis', aspect='auto')
    axes[1].set_title(f'{corruption} (severity {severity}) Distances', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(0, num_classes, 5))
    axes[1].set_yticks(range(0, num_classes, 5))
    axes[1].set_xticklabels(range(0, num_classes, 5), fontsize=8)
    axes[1].set_yticklabels(range(0, num_classes, 5), fontsize=8)
    axes[1].set_xlabel('Class Index')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Distance change (corrupted - clean)
    distance_change = corrupted_distances - clean_distances
    vmax = np.abs(distance_change).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im3 = axes[2].imshow(distance_change, cmap='RdBu_r', norm=norm, aspect='auto')
    axes[2].set_title('Distance Change\n(Red = Closer, Blue = Further)', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(0, num_classes, 5))
    axes[2].set_yticks(range(0, num_classes, 5))
    axes[2].set_xticklabels(range(0, num_classes, 5), fontsize=8)
    axes[2].set_yticklabels(range(0, num_classes, 5), fontsize=8)
    axes[2].set_xlabel('Class Index')
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.suptitle(f'Class Distance Matrices in Attention Space\n{corruption} corruption, severity {severity}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_top_confused_pairs(
    analysis: Dict,
    corruption: str,
    severity: int,
    save_path: str
):
    """
    Visualize the top confused class pairs.
    """
    top_pairs = analysis['top_confused_pairs']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    pair_labels = [f"{p['class_i']}\n↔\n{p['class_j']}" for p in top_pairs]
    clean_dists = [p['clean_distance'] for p in top_pairs]
    corrupted_dists = [p['corrupted_distance'] for p in top_pairs]

    x = np.arange(len(pair_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, clean_dists, width, label='Clean', color='steelblue')
    bars2 = ax.bar(x + width/2, corrupted_dists, width, label='Corrupted', color='coral')

    ax.set_ylabel('Cosine Distance', fontsize=11)
    ax.set_title(f'Top {len(top_pairs)} Most Confused Class Pairs\n{corruption} severity {severity}',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=8, rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add distance change annotations
    for i, pair in enumerate(top_pairs):
        change = pair['distance_change']
        ax.annotate(f'{change:+.3f}',
                   xy=(i, max(clean_dists[i], corrupted_dists[i]) + 0.01),
                   ha='center', va='bottom', fontsize=8, color='red' if change < 0 else 'green')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_tsne_with_displacement(
    clean_centroids: np.ndarray,
    corrupted_centroids: np.ndarray,
    class_names: List[str],
    corruption: str,
    severity: int,
    save_path: str
):
    """
    Visualize class centroids using t-SNE with displacement arrows.
    """
    if not SKLEARN_AVAILABLE:
        logging.warning("sklearn not available. Skipping t-SNE visualization.")
        return

    num_classes = len(class_names)

    # Combine all centroids for joint t-SNE
    all_centroids = np.vstack([clean_centroids, corrupted_centroids])

    # Apply t-SNE
    # Note: 'n_iter' was renamed to 'max_iter' in sklearn >= 1.5
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, num_classes - 1),
                    random_state=42, max_iter=1000)
    except TypeError:
        # Fallback for older sklearn versions
        tsne = TSNE(n_components=2, perplexity=min(30, num_classes - 1),
                    random_state=42, n_iter=1000)
    embedded = tsne.fit_transform(all_centroids)

    clean_embedded = embedded[:num_classes]
    corrupted_embedded = embedded[num_classes:]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Color map for classes
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_classes)))
    if num_classes > 20:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

    # Plot clean centroids (circles)
    for i in range(num_classes):
        ax.scatter(clean_embedded[i, 0], clean_embedded[i, 1],
                  c=[colors[i % len(colors)]], s=100, marker='o',
                  edgecolors='black', linewidth=1, alpha=0.8)

    # Plot corrupted centroids (triangles)
    for i in range(num_classes):
        ax.scatter(corrupted_embedded[i, 0], corrupted_embedded[i, 1],
                  c=[colors[i % len(colors)]], s=100, marker='^',
                  edgecolors='black', linewidth=1, alpha=0.8)

    # Draw displacement arrows
    for i in range(num_classes):
        dx = corrupted_embedded[i, 0] - clean_embedded[i, 0]
        dy = corrupted_embedded[i, 1] - clean_embedded[i, 1]
        ax.annotate('', xy=(corrupted_embedded[i, 0], corrupted_embedded[i, 1]),
                   xytext=(clean_embedded[i, 0], clean_embedded[i, 1]),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1))

    # Add class labels at clean positions
    for i in range(num_classes):
        ax.annotate(class_names[i], (clean_embedded[i, 0], clean_embedded[i, 1]),
                   fontsize=7, ha='center', va='bottom', alpha=0.8)

    # Legend
    ax.scatter([], [], c='gray', s=100, marker='o', label='Clean')
    ax.scatter([], [], c='gray', s=100, marker='^', label='Corrupted')
    ax.legend(loc='upper right')

    ax.set_title(f't-SNE of Class Attention Centroids\n{corruption} severity {severity}\n(Arrows show displacement from clean → corrupted)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_displacement_magnitudes(
    clean_centroids: np.ndarray,
    corrupted_centroids: np.ndarray,
    class_names: List[str],
    corruption: str,
    severity: int,
    save_path: str
):
    """
    Visualize displacement magnitude for each class.
    """
    # Compute displacement in original attention space
    displacements = np.linalg.norm(corrupted_centroids - clean_centroids, axis=1)

    # Sort by displacement
    sorted_indices = np.argsort(displacements)[::-1]

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.RdYlGn_r(displacements[sorted_indices] / displacements.max())

    bars = ax.barh(range(len(class_names)), displacements[sorted_indices], color=colors)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels([class_names[i] for i in sorted_indices], fontsize=9)
    ax.set_xlabel('Displacement Magnitude (L2 norm in attention space)', fontsize=11)
    ax.set_title(f'Class Displacement Under {corruption} Corruption (severity {severity})\n(Higher = More affected by corruption)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add values on bars
    for i, (idx, bar) in enumerate(zip(sorted_indices, bars)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{displacements[idx]:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_severity_progression(
    all_results: Dict[int, Dict],
    class_names: List[str],
    corruption: str,
    save_path: str
):
    """
    Visualize how confusion metrics change across severity levels.
    """
    severities = sorted(all_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean distance change across severities
    ax1 = axes[0, 0]
    mean_changes = [all_results[s]['analysis']['mean_distance_change'] for s in severities]
    ax1.plot(severities, mean_changes, 'o-', color='coral', linewidth=2, markersize=8)
    ax1.set_xlabel('Severity')
    ax1.set_ylabel('Mean Distance Change')
    ax1.set_title('Mean Distance Change Across Severities\n(Negative = Classes Getting Closer)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # 2. Neighbor change ratio across severities
    ax2 = axes[0, 1]
    neighbor_ratios = [all_results[s]['analysis']['neighbor_change_ratio'] for s in severities]
    ax2.plot(severities, neighbor_ratios, 's-', color='steelblue', linewidth=2, markersize=8)
    ax2.set_xlabel('Severity')
    ax2.set_ylabel('Ratio of Classes with Changed Nearest Neighbor')
    ax2.set_title('Nearest Neighbor Instability Across Severities', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # 3. Top confused pair distances across severities
    ax3 = axes[1, 0]
    # Get the top confused pair from severity 5
    top_pair = all_results[5]['analysis']['top_confused_pairs'][0]
    i, j = top_pair['class_i_idx'], top_pair['class_j_idx']
    pair_label = f"{top_pair['class_i']} ↔ {top_pair['class_j']}"

    clean_dists = [all_results[s]['clean_distances'][i, j] for s in severities]
    corrupted_dists = [all_results[s]['corrupted_distances'][i, j] for s in severities]

    ax3.plot(severities, clean_dists, 'o--', label='Clean', color='green', alpha=0.7)
    ax3.plot(severities, corrupted_dists, 's-', label='Corrupted', color='red', linewidth=2)
    ax3.set_xlabel('Severity')
    ax3.set_ylabel('Cosine Distance')
    ax3.set_title(f'Most Confused Pair: {pair_label}\n(Distance trend)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Heatmap of confusion change across severities for top pairs
    ax4 = axes[1, 1]
    top_k = 10
    top_pairs = all_results[5]['analysis']['top_confused_pairs'][:top_k]

    change_matrix = np.zeros((top_k, len(severities)))
    for col, s in enumerate(severities):
        for row, pair in enumerate(top_pairs):
            i, j = pair['class_i_idx'], pair['class_j_idx']
            change = all_results[s]['corrupted_distances'][i, j] - all_results[s]['clean_distances'][i, j]
            change_matrix[row, col] = change

    im = ax4.imshow(change_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(change_matrix).max(), vmax=np.abs(change_matrix).max())
    ax4.set_xticks(range(len(severities)))
    ax4.set_xticklabels(severities)
    ax4.set_yticks(range(top_k))
    ax4.set_yticklabels([f"{p['class_i'][:8]}↔{p['class_j'][:8]}" for p in top_pairs], fontsize=8)
    ax4.set_xlabel('Severity')
    ax4.set_title('Distance Change for Top Confused Pairs\n(Red = Closer)', fontweight='bold')
    plt.colorbar(im, ax=ax4, fraction=0.046)

    plt.suptitle(f'{corruption} Corruption: Severity Progression Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run cross-class attention analysis."""
    # Parse arguments
    args = parse_args()

    # Add custom arguments for this script
    import sys

    # Check for samples-per-class argument
    samples_per_class = None
    for i, arg in enumerate(sys.argv):
        if arg == '--samples-per-class' and i + 1 < len(sys.argv):
            samples_per_class = int(sys.argv[i + 1])
            break

    if samples_per_class is None:
        samples_per_class = 20  # Default

    # Set device
    if not hasattr(args, 'device') or args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    corruption = args.corruption
    if corruption == 'clean' or corruption == 'all':
        logging.error("Please specify a specific corruption type (e.g., --corruption cutout)")
        return

    logging.info("=" * 70)
    logging.info("Cross-Class Attention Analysis")
    logging.info("=" * 70)
    logging.info(f"Corruption: {corruption}")
    logging.info(f"Samples per class: {samples_per_class}")
    logging.info(f"Device: {args.device}")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f'cross_class_analysis_{corruption}')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    logging.info("\nLoading Uni3D model...")
    _, model = load_vlm_model(args)
    model.eval()

    # Initialize analyzer
    analyzer = CrossClassAttentionAnalyzer(model, device=args.device)

    # Store results for all severities
    all_results = {}

    severities = [1, 2, 3, 4, 5]

    for severity in severities:
        logging.info(f"\n{'='*50}")
        logging.info(f"Processing Severity {severity}")
        logging.info(f"{'='*50}")

        severity_dir = os.path.join(output_dir, f'severity_{severity}')
        os.makedirs(severity_dir, exist_ok=True)

        # Extract clean attention vectors
        logging.info("\nExtracting clean attention vectors...")
        clean_attention, class_names = analyzer.extract_all_attention_vectors(
            args, corruption='clean', severity=1, samples_per_class=samples_per_class
        )

        # Extract corrupted attention vectors
        logging.info(f"\nExtracting {corruption} attention vectors (severity {severity})...")
        corrupted_attention, _ = analyzer.extract_all_attention_vectors(
            args, corruption=corruption, severity=severity, samples_per_class=samples_per_class
        )

        # Compute centroids
        logging.info("\nComputing class centroids...")
        clean_centroids = analyzer.compute_class_centroids(clean_attention)
        corrupted_centroids = analyzer.compute_class_centroids(corrupted_attention)

        # Compute distance matrices
        logging.info("Computing distance matrices...")
        clean_distances = analyzer.compute_distance_matrix(clean_centroids)
        corrupted_distances = analyzer.compute_distance_matrix(corrupted_centroids)

        # Analyze confusion
        logging.info("Analyzing class confusion...")
        analysis = analyzer.analyze_confusion(clean_distances, corrupted_distances, class_names)

        # Store results
        all_results[severity] = {
            'clean_centroids': clean_centroids,
            'corrupted_centroids': corrupted_centroids,
            'clean_distances': clean_distances,
            'corrupted_distances': corrupted_distances,
            'analysis': analysis,
            'class_names': class_names
        }

        # Generate visualizations
        logging.info("Generating visualizations...")

        # Distance matrices
        visualize_distance_matrices(
            clean_distances, corrupted_distances, class_names,
            corruption, severity,
            os.path.join(severity_dir, 'distance_matrices.png')
        )

        # Top confused pairs
        visualize_top_confused_pairs(
            analysis, corruption, severity,
            os.path.join(severity_dir, 'top_confused_pairs.png')
        )

        # t-SNE with displacement
        visualize_tsne_with_displacement(
            clean_centroids, corrupted_centroids, class_names,
            corruption, severity,
            os.path.join(severity_dir, 'tsne_displacement.png')
        )

        # Displacement magnitudes
        visualize_displacement_magnitudes(
            clean_centroids, corrupted_centroids, class_names,
            corruption, severity,
            os.path.join(severity_dir, 'displacement_magnitudes.png')
        )

        # Save analysis to JSON
        def convert_to_serializable(v):
            """Convert numpy types to Python native types for JSON serialization."""
            # Check bool first (bool is subclass of int in Python)
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            elif isinstance(v, (np.floating, float)):
                return float(v)
            elif isinstance(v, (np.integer, int)):
                return int(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            else:
                return v

        analysis_json = {
            'corruption': corruption,
            'severity': severity,
            'num_classes': len(class_names),
            'samples_per_class': samples_per_class,
            'mean_distance_change': float(analysis['mean_distance_change']),
            'neighbor_change_ratio': float(analysis['neighbor_change_ratio']),
            'top_confused_pairs': [
                {k: convert_to_serializable(v) for k, v in pair.items()}
                for pair in analysis['top_confused_pairs']
            ],
            'class_movement': [
                {k: convert_to_serializable(v) for k, v in movement.items()}
                for movement in analysis['class_movement']
            ]
        }

        with open(os.path.join(severity_dir, 'analysis.json'), 'w') as f:
            json.dump(analysis_json, f, indent=2)

        # Print top confused pairs
        logging.info(f"\nTop 5 Most Confused Pairs (severity {severity}):")
        for i, pair in enumerate(analysis['top_confused_pairs'][:5]):
            logging.info(f"  {i+1}. {pair['class_i']} ↔ {pair['class_j']}: "
                        f"change = {pair['distance_change']:.4f} "
                        f"(clean: {pair['clean_distance']:.4f} → corrupted: {pair['corrupted_distance']:.4f})")

        # Save centroids as numpy arrays
        np.save(os.path.join(severity_dir, 'clean_centroids.npy'), clean_centroids)
        np.save(os.path.join(severity_dir, 'corrupted_centroids.npy'), corrupted_centroids)
        np.save(os.path.join(severity_dir, 'clean_distances.npy'), clean_distances)
        np.save(os.path.join(severity_dir, 'corrupted_distances.npy'), corrupted_distances)

    # Generate severity progression visualization
    logging.info("\nGenerating severity progression visualization...")
    visualize_severity_progression(
        all_results, class_names, corruption,
        os.path.join(output_dir, 'severity_progression.png')
    )

    # Summary across all severities
    logging.info("\n" + "=" * 70)
    logging.info("SUMMARY ACROSS ALL SEVERITIES")
    logging.info("=" * 70)

    for severity in severities:
        analysis = all_results[severity]['analysis']
        logging.info(f"\nSeverity {severity}:")
        logging.info(f"  Mean distance change: {analysis['mean_distance_change']:.4f}")
        logging.info(f"  Classes with changed nearest neighbor: {analysis['neighbor_change_ratio']*100:.1f}%")
        top_pair = analysis['top_confused_pairs'][0]
        logging.info(f"  Most confused pair: {top_pair['class_i']} ↔ {top_pair['class_j']} "
                    f"(change: {top_pair['distance_change']:.4f})")

    # Cleanup
    analyzer.extractor.remove_hooks()

    logging.info(f"\n{'='*70}")
    logging.info(f"Analysis complete! Results saved to: {output_dir}")
    logging.info(f"{'='*70}")


if __name__ == '__main__':
    main()
