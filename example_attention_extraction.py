"""
Example: Extracting and Visualizing Attention Maps from Uni3D

This script shows how to:
1. Load a Uni3D model
2. Load a sample from the dataset (clean and corrupted versions)
3. Extract attention maps from both versions
4. Visualize and compare attention patterns

Usage:
    # Run with default settings (uses args from utils/params.py)
    python example_attention_extraction.py --myroot /path/to/dataset --corruption cutout --severity 5

    # Compare clean vs corrupted attention
    python example_attention_extraction.py --myroot /path/to/modelnet40_c --corruption jitter --severity 3
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import random
import copy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import from project
from utils.params import parse_args
from utils.load_models import load_vlm_model
from data.data_utils import load_tta_dataset
from extract_attention import (
    AttentionExtractor,
    visualize_attention_maps,
    visualize_attention_averaged_over_heads,
    visualize_cls_attention_across_layers,
    visualize_layer_attention_on_pointcloud_grid,
    visualize_attention_on_pointcloud,
    visualize_attention_heads_on_pointcloud,
    visualize_attention_evolution,
    plot_attention_statistics
)


def load_sample_pair(args, sample_idx=None):
    """
    Load the same sample in both clean and corrupted versions.

    Args:
        args: Parsed arguments
        sample_idx: Index of sample to load (None for random)

    Returns:
        clean_data: (pc, label, class_name, rgb) for clean sample
        corrupted_data: (pc, label, class_name, rgb) for corrupted sample
        sample_idx: The index used
    """
    # Load corrupted dataset first to get length
    corrupted_dataset = load_tta_dataset(args)

    # Select random sample if not specified
    if sample_idx is None:
        sample_idx = random.randint(0, len(corrupted_dataset) - 1)

    logging.info(f"Selected sample index: {sample_idx}")

    # Get corrupted sample
    corrupted_data = corrupted_dataset[sample_idx]

    # Load clean version
    args_clean = copy.deepcopy(args)
    args_clean.corruption = 'clean'
    clean_dataset = load_tta_dataset(args_clean)
    clean_data = clean_dataset[sample_idx]

    return clean_data, corrupted_data, sample_idx


def extract_attention_from_sample(model, extractor, pc, rgb, device):
    """
    Extract attention maps from a point cloud sample.

    Args:
        model: Uni3D model
        extractor: AttentionExtractor instance
        pc: Point cloud numpy array [N, 3]
        rgb: RGB numpy array [N, 3]
        device: Device to run on

    Returns:
        attention_maps: Dictionary of attention maps
        group_centers: Group centers numpy array
    """
    # Convert to tensors
    pc_tensor = torch.from_numpy(pc).float().unsqueeze(0).to(device)

    # Handle rgb - could be tensor or numpy
    if isinstance(rgb, torch.Tensor):
        rgb_np = rgb.numpy()
    else:
        rgb_np = rgb
    rgb_tensor = torch.from_numpy(rgb_np).float().unsqueeze(0).to(device)

    # Combine XYZ and RGB
    feature = torch.cat([pc_tensor, rgb_tensor], dim=-1)

    # Extract attention
    attention_maps = extractor.extract(feature)

    # Get group centers
    group_centers = extractor.get_group_centers(feature)
    group_centers_np = group_centers[0].numpy()

    return attention_maps, group_centers_np


def visualize_comparison(
    clean_attention, corrupted_attention,
    clean_centers, corrupted_centers,
    clean_pc, corrupted_pc,
    class_name, corruption_type, severity,
    output_dir
):
    """
    Generate comparison visualizations between clean and corrupted attention.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Side-by-side CLS attention evolution
    logging.info("Generating CLS attention evolution comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Get layer indices
    layer_indices = sorted([int(k.split('_')[1]) for k in clean_attention.keys()])
    n_layers = len(layer_indices)

    # Clean attention matrix
    clean_cls_matrix = []
    for layer_idx in layer_indices:
        attn = clean_attention[f'layer_{layer_idx}'][0]
        cls_attn = attn.mean(dim=0)[0, 1:].numpy()
        clean_cls_matrix.append(cls_attn)
    clean_cls_matrix = np.array(clean_cls_matrix)

    # Corrupted attention matrix
    corrupted_cls_matrix = []
    for layer_idx in layer_indices:
        attn = corrupted_attention[f'layer_{layer_idx}'][0]
        cls_attn = attn.mean(dim=0)[0, 1:].numpy()
        corrupted_cls_matrix.append(cls_attn)
    corrupted_cls_matrix = np.array(corrupted_cls_matrix)

    # Plot clean
    im1 = axes[0].imshow(clean_cls_matrix, aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Token Index', fontsize=11)
    axes[0].set_ylabel('Layer', fontsize=11)
    axes[0].set_title(f'Clean - {class_name}', fontsize=12, fontweight='bold')
    axes[0].set_yticks(range(n_layers))
    axes[0].set_yticklabels(layer_indices)
    plt.colorbar(im1, ax=axes[0])

    # Plot corrupted
    im2 = axes[1].imshow(corrupted_cls_matrix, aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Token Index', fontsize=11)
    axes[1].set_ylabel('Layer', fontsize=11)
    axes[1].set_title(f'{corruption_type} (severity {severity}) - {class_name}', fontsize=12, fontweight='bold')
    axes[1].set_yticks(range(n_layers))
    axes[1].set_yticklabels(layer_indices)
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle('CLS Attention Evolution: Clean vs Corrupted\n(Averaged over heads)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_comparison_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Attention difference heatmap
    logging.info("Generating attention difference heatmap...")
    diff_matrix = corrupted_cls_matrix - clean_cls_matrix

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(diff_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-np.abs(diff_matrix).max(), vmax=np.abs(diff_matrix).max())
    ax.set_xlabel('Token Index', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title(f'Attention Difference (Corrupted - Clean)\n{corruption_type} severity {severity} | Class: {class_name}',
                 fontsize=12, fontweight='bold')
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_indices)
    plt.colorbar(im, ax=ax, label='Attention Difference')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_difference.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Statistics comparison across layers
    logging.info("Generating statistics comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Compute statistics for both
    clean_stats = {'entropy': [], 'max': [], 'sparsity': []}
    corrupted_stats = {'entropy': [], 'max': [], 'sparsity': []}

    for i, layer_idx in enumerate(layer_indices):
        # Clean
        cls_attn = clean_cls_matrix[i]
        clean_stats['entropy'].append(-np.sum(cls_attn * np.log(cls_attn + 1e-10)))
        clean_stats['max'].append(cls_attn.max())
        k = max(1, len(cls_attn) // 10)
        clean_stats['sparsity'].append(np.sort(cls_attn)[-k:].sum())

        # Corrupted
        cls_attn = corrupted_cls_matrix[i]
        corrupted_stats['entropy'].append(-np.sum(cls_attn * np.log(cls_attn + 1e-10)))
        corrupted_stats['max'].append(cls_attn.max())
        corrupted_stats['sparsity'].append(np.sort(cls_attn)[-k:].sum())

    # Plot entropy
    axes[0, 0].plot(layer_indices, clean_stats['entropy'], 'o-', label='Clean', color='blue')
    axes[0, 0].plot(layer_indices, corrupted_stats['entropy'], 's-', label='Corrupted', color='red')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Entropy')
    axes[0, 0].set_title('Attention Entropy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot max attention
    axes[0, 1].plot(layer_indices, clean_stats['max'], 'o-', label='Clean', color='blue')
    axes[0, 1].plot(layer_indices, corrupted_stats['max'], 's-', label='Corrupted', color='red')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Max Attention')
    axes[0, 1].set_title('Maximum Attention Weight')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot sparsity
    axes[1, 0].plot(layer_indices, clean_stats['sparsity'], 'o-', label='Clean', color='blue')
    axes[1, 0].plot(layer_indices, corrupted_stats['sparsity'], 's-', label='Corrupted', color='red')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Top-10% Mass')
    axes[1, 0].set_title('Attention Sparsity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot cosine similarity between clean and corrupted at each layer
    similarities = []
    for i in range(len(layer_indices)):
        a = clean_cls_matrix[i]
        b = corrupted_cls_matrix[i]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        similarities.append(sim)

    axes[1, 1].bar(layer_indices, similarities, color='teal', alpha=0.7)
    axes[1, 1].axhline(y=np.mean(similarities), color='red', linestyle='--',
                       label=f'Mean: {np.mean(similarities):.3f}')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Cosine Similarity')
    axes[1, 1].set_title('Clean vs Corrupted Attention Similarity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Attention Statistics: Clean vs {corruption_type}\nClass: {class_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_statistics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Individual visualizations for clean
    logging.info("Generating clean sample visualizations...")
    clean_dir = os.path.join(output_dir, 'clean')
    os.makedirs(clean_dir, exist_ok=True)

    visualize_attention_averaged_over_heads(
        clean_attention,
        save_path=os.path.join(clean_dir, 'attention_averaged.png')
    )

    visualize_cls_attention_across_layers(
        clean_attention,
        save_path=os.path.join(clean_dir, 'cls_attention_evolution.png')
    )

    plot_attention_statistics(
        clean_attention,
        save_path=os.path.join(clean_dir, 'attention_stats.png')
    )

    # 5. Individual visualizations for corrupted
    logging.info("Generating corrupted sample visualizations...")
    corrupted_dir = os.path.join(output_dir, 'corrupted')
    os.makedirs(corrupted_dir, exist_ok=True)

    visualize_attention_averaged_over_heads(
        corrupted_attention,
        save_path=os.path.join(corrupted_dir, 'attention_averaged.png')
    )

    visualize_cls_attention_across_layers(
        corrupted_attention,
        save_path=os.path.join(corrupted_dir, 'cls_attention_evolution.png')
    )

    plot_attention_statistics(
        corrupted_attention,
        save_path=os.path.join(corrupted_dir, 'attention_stats.png')
    )

    # 6. 3D visualizations (if plotly available)
    try:
        import plotly
        logging.info("Generating 3D visualizations...")

        # Get last layer attention
        last_layer_idx = max(layer_indices)

        # Clean 3D
        attn_clean_last = clean_attention[f'layer_{last_layer_idx}'][0, :, 0, 1:].mean(dim=0).numpy()
        visualize_attention_on_pointcloud(
            clean_pc, attn_clean_last, clean_centers,
            title=f'Clean Attention - {class_name} (Layer {last_layer_idx})',
            save_path=os.path.join(clean_dir, 'attention_3d.html')
        )

        visualize_layer_attention_on_pointcloud_grid(
            clean_attention, clean_pc, clean_centers,
            save_path=os.path.join(clean_dir, 'attention_layers_grid.html')
        )

        # Corrupted 3D
        attn_corrupted_last = corrupted_attention[f'layer_{last_layer_idx}'][0, :, 0, 1:].mean(dim=0).numpy()
        visualize_attention_on_pointcloud(
            corrupted_pc, attn_corrupted_last, corrupted_centers,
            title=f'{corruption_type} Attention - {class_name} (Layer {last_layer_idx})',
            save_path=os.path.join(corrupted_dir, 'attention_3d.html')
        )

        visualize_layer_attention_on_pointcloud_grid(
            corrupted_attention, corrupted_pc, corrupted_centers,
            save_path=os.path.join(corrupted_dir, 'attention_layers_grid.html')
        )

    except ImportError:
        logging.warning("Plotly not installed. Skipping 3D visualizations.")

    logging.info(f"\nAll visualizations saved to: {output_dir}")


def main():
    """Main function to run attention extraction and visualization."""
    # Parse arguments from utils/params.py
    args = parse_args()

    # Set device
    if not hasattr(args, 'device') or args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info("=" * 60)
    logging.info("Uni3D Attention Extraction: Clean vs Corrupted Comparison")
    logging.info("=" * 60)
    logging.info(f"Dataset: {args.dataset_name}")
    logging.info(f"Corruption: {args.corruption}")
    logging.info(f"Severity: {args.severity}")
    logging.info(f"Device: {args.device}")

    # Create output directory
    output_dir = os.path.join(args.output_dir, f'attention_vis_{args.corruption}_s{args.severity}')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    logging.info("\nLoading Uni3D model...")
    _, model = load_vlm_model(args)
    model.eval()

    # Initialize extractor
    extractor = AttentionExtractor(model, device=args.device)

    # Load sample pair (clean and corrupted)
    logging.info("\nLoading dataset samples...")
    try:
        clean_data, corrupted_data, sample_idx = load_sample_pair(args)

        clean_pc, clean_label, clean_class_name, clean_rgb = clean_data
        corrupted_pc, corrupted_label, corrupted_class_name, corrupted_rgb = corrupted_data

        logging.info(f"Sample class: {clean_class_name} (label: {clean_label})")
        logging.info(f"Clean PC shape: {clean_pc.shape}")
        logging.info(f"Corrupted PC shape: {corrupted_pc.shape}")

        # Extract attention from clean sample
        logging.info("\nExtracting attention from clean sample...")
        clean_attention, clean_centers = extract_attention_from_sample(
            model, extractor, clean_pc, clean_rgb, args.device
        )
        logging.info(f"Extracted {len(clean_attention)} layers")

        # Extract attention from corrupted sample
        logging.info(f"\nExtracting attention from {args.corruption} sample...")
        corrupted_attention, corrupted_centers = extract_attention_from_sample(
            model, extractor, corrupted_pc, corrupted_rgb, args.device
        )
        logging.info(f"Extracted {len(corrupted_attention)} layers")

        # Generate comparison visualizations
        logging.info("\nGenerating visualizations...")
        visualize_comparison(
            clean_attention, corrupted_attention,
            clean_centers, corrupted_centers,
            clean_pc, corrupted_pc,
            clean_class_name, args.corruption, args.severity,
            output_dir
        )

        # Save sample info
        with open(os.path.join(output_dir, 'sample_info.txt'), 'w') as f:
            f.write(f"Sample Index: {sample_idx}\n")
            f.write(f"Class: {clean_class_name}\n")
            f.write(f"Label: {clean_label}\n")
            f.write(f"Corruption: {args.corruption}\n")
            f.write(f"Severity: {args.severity}\n")
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Root: {args.myroot}\n")

    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        logging.info("\nFalling back to synthetic data...")

        # Create synthetic point cloud
        n_points = 1024
        np.random.seed(args.seed)

        # Create a sphere
        phi = np.random.uniform(0, 2*np.pi, n_points)
        theta = np.random.uniform(0, np.pi, n_points)
        r = 1.0

        clean_pc = np.stack([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ], axis=1).astype(np.float32)
        clean_rgb = np.ones_like(clean_pc)

        # Create corrupted version (add noise)
        corrupted_pc = clean_pc + np.random.normal(0, 0.05, clean_pc.shape).astype(np.float32)
        corrupted_rgb = clean_rgb

        logging.info(f"Created synthetic sphere with {n_points} points")

        # Extract attention
        logging.info("\nExtracting attention from clean synthetic sample...")
        clean_attention, clean_centers = extract_attention_from_sample(
            model, extractor, clean_pc, clean_rgb, args.device
        )

        logging.info("\nExtracting attention from noisy synthetic sample...")
        corrupted_attention, corrupted_centers = extract_attention_from_sample(
            model, extractor, corrupted_pc, corrupted_rgb, args.device
        )

        # Generate visualizations
        visualize_comparison(
            clean_attention, corrupted_attention,
            clean_centers, corrupted_centers,
            clean_pc, corrupted_pc,
            'synthetic_sphere', 'noise', 1,
            output_dir
        )

    # Cleanup
    extractor.remove_hooks()

    logging.info("\n" + "=" * 60)
    logging.info("Attention extraction and visualization complete!")
    logging.info(f"Results saved to: {output_dir}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
