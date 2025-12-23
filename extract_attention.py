"""
Attention Map Extraction and Visualization for Uni3D

This script extracts attention maps from the Uni3D point cloud encoder
and visualizes them for different layers and attention heads.

Usage:
    python extract_attention.py --pc-path /path/to/pointcloud.npy

Or import and use programmatically:
    from extract_attention import AttentionExtractor, visualize_attention_maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from typing import Dict, List, Tuple, Optional
import logging

# For 3D visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. 3D visualizations will be disabled.")


class AttentionExtractor:
    """
    Extracts attention maps from Uni3D's transformer blocks.

    The Uni3D model uses a timm-based Vision Transformer (EVA02)
    where each block has an attention layer with multiple heads.

    Architecture:
        Point Cloud → Group (FPS) → Encoder → [CLS + Tokens] → Transformer Blocks → Output
                                                                    ↑
                                                            Attention here!
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Initialize the attention extractor.

        Args:
            model: The Uni3D model (or just the point_encoder)
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.hooks: List = []

        # Get the transformer blocks
        if hasattr(model, 'point_encoder'):
            self.transformer = model.point_encoder.visual
            self.point_encoder = model.point_encoder
        else:
            self.transformer = model.visual
            self.point_encoder = model

        # Identify attention layers
        self.num_layers = len(self.transformer.blocks)
        self.num_heads = self._get_num_heads()

        logging.info(f"AttentionExtractor initialized:")
        logging.info(f"  - Number of transformer layers: {self.num_layers}")
        logging.info(f"  - Number of attention heads: {self.num_heads}")

    def _get_num_heads(self) -> int:
        """Get the number of attention heads from the first block."""
        first_block = self.transformer.blocks[0]
        if hasattr(first_block, 'attn'):
            attn = first_block.attn
            if hasattr(attn, 'num_heads'):
                return attn.num_heads
        # Default for EVA02-Large
        return 16

    def _attention_hook(self, layer_idx: int):
        """
        Create a hook function for a specific layer.

        The hook captures attention weights from the attention layer.
        """
        def hook(module, input, output):
            # For timm's attention, we need to manually compute attention weights
            # since they don't store them by default

            # Get the attention module
            if hasattr(module, 'qkv'):
                # Standard timm attention
                B, N, C = input[0].shape
                qkv = module.qkv(input[0])
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, head_dim
                q, k, v = qkv[0], qkv[1], qkv[2]

                # Compute attention weights
                scale = (C // module.num_heads) ** -0.5
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)

                # Store attention map: [B, num_heads, N, N]
                self.attention_maps[f'layer_{layer_idx}'] = attn.detach().cpu()

        return hook

    def _attention_hook_with_save(self, layer_idx: int):
        """
        Alternative hook that captures attention by modifying forward pass.
        Works with models that use scaled_dot_product_attention.
        Handles both standard qkv and EVA-style separate q_proj/k_proj/v_proj.
        """
        def hook(module, input, output):
            # Store input for manual attention computation
            x = input[0]
            B, N, C = x.shape
            num_heads = module.num_heads

            # Check for EVA-style separate projections (qkv is None)
            if hasattr(module, 'q_proj') and module.q_proj is not None:
                # EVA attention with separate q, k, v projections
                q = module.q_proj(x)
                k = module.k_proj(x)
                v = module.v_proj(x)

                # Add biases if present
                if hasattr(module, 'q_bias') and module.q_bias is not None:
                    q = q + module.q_bias
                if hasattr(module, 'k_bias') and module.k_bias is not None:
                    k = k + module.k_bias

                head_dim = q.shape[-1] // num_heads

                # Reshape to [B, N, num_heads, head_dim] then [B, num_heads, N, head_dim]
                q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
                k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

                # Apply normalization if present (EVA uses q_norm, k_norm)
                if hasattr(module, 'q_norm') and module.q_norm is not None:
                    q = module.q_norm(q)
                if hasattr(module, 'k_norm') and module.k_norm is not None:
                    k = module.k_norm(k)

                # Use scale from module if available, else compute
                scale = getattr(module, 'scale', head_dim ** -0.5)

                # Manual attention computation
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_weights, dim=-1)

                self.attention_maps[f'layer_{layer_idx}'] = attn_weights.detach().cpu()

            elif hasattr(module, 'qkv') and module.qkv is not None:
                # Standard timm attention with combined qkv
                qkv = module.qkv(x)
                head_dim = C // num_heads

                qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                # Manual attention computation
                scale = head_dim ** -0.5
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = F.softmax(attn_weights, dim=-1)

                self.attention_maps[f'layer_{layer_idx}'] = attn_weights.detach().cpu()

        return hook

    def register_hooks(self):
        """Register forward hooks on all attention layers."""
        self.remove_hooks()  # Clear any existing hooks

        for idx, block in enumerate(self.transformer.blocks):
            if hasattr(block, 'attn'):
                hook = block.attn.register_forward_hook(
                    self._attention_hook_with_save(idx)
                )
                self.hooks.append(hook)

        logging.info(f"Registered {len(self.hooks)} attention hooks")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_maps = {}

    @torch.no_grad()
    def extract(self, point_cloud: torch.Tensor, rgb: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for a given point cloud.

        Args:
            point_cloud: Point cloud tensor [B, N, 3] or [B, N, 6] (with RGB)
            rgb: Optional RGB tensor [B, N, 3]. If None, uses ones.

        Returns:
            Dictionary mapping layer names to attention tensors
            Each tensor has shape [B, num_heads, seq_len, seq_len]
            where seq_len = num_groups + 1 (for CLS token)
        """
        self.model.eval()
        self.attention_maps = {}

        # Prepare input
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)

        point_cloud = point_cloud.to(self.device)

        # Handle RGB
        if point_cloud.shape[-1] == 6:
            xyz = point_cloud[:, :, :3]
            rgb = point_cloud[:, :, 3:]
        else:
            xyz = point_cloud[:, :, :3]
            if rgb is None:
                rgb = torch.ones_like(xyz)
            rgb = rgb.to(self.device)

        # Combine for model input
        feature = torch.cat((xyz, rgb), dim=-1)

        # Register hooks
        self.register_hooks()

        try:
            # Forward pass
            if hasattr(self.model, 'encode_pc'):
                _ = self.model.encode_pc(feature)
            else:
                _ = self.point_encoder(xyz, rgb)
        finally:
            # Always remove hooks after extraction
            pass  # Keep hooks for potential re-use

        return self.attention_maps.copy()

    def get_attention_to_cls(self, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention FROM all tokens TO the CLS token.

        This shows which point groups the model attends to when forming
        the final representation.

        Args:
            layer_idx: Which layer to get (-1 for last layer)

        Returns:
            Tensor of shape [B, num_heads, num_groups]
        """
        if layer_idx == -1:
            layer_idx = self.num_layers - 1

        key = f'layer_{layer_idx}'
        if key not in self.attention_maps:
            raise ValueError(f"Layer {layer_idx} attention not found. Run extract() first.")

        attn = self.attention_maps[key]  # [B, heads, seq_len, seq_len]
        # Attention TO cls (first token) FROM all other tokens
        attn_to_cls = attn[:, :, 0, 1:]  # [B, heads, num_groups]
        return attn_to_cls

    def get_attention_from_cls(self, layer_idx: int = -1) -> torch.Tensor:
        """
        Get attention FROM the CLS token TO all other tokens.

        Args:
            layer_idx: Which layer to get (-1 for last layer)

        Returns:
            Tensor of shape [B, num_heads, num_groups]
        """
        if layer_idx == -1:
            layer_idx = self.num_layers - 1

        key = f'layer_{layer_idx}'
        if key not in self.attention_maps:
            raise ValueError(f"Layer {layer_idx} attention not found. Run extract() first.")

        attn = self.attention_maps[key]  # [B, heads, seq_len, seq_len]
        # Attention FROM cls (row 0) TO all tokens (exclude self)
        attn_from_cls = attn[:, :, 0, 1:]  # [B, heads, num_groups]
        return attn_from_cls

    def get_group_centers(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Get the center coordinates of each point group.

        These correspond to the tokens in the attention map.

        Args:
            point_cloud: Input point cloud [B, N, 3+]

        Returns:
            Group centers [B, num_groups, 3]
        """
        point_cloud = point_cloud.to(self.device)
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)

        xyz = point_cloud[:, :, :3].contiguous()
        rgb = point_cloud[:, :, 3:6] if point_cloud.shape[-1] >= 6 else torch.ones_like(xyz)

        # Use the group divider
        _, centers, _ = self.point_encoder.group_divider(xyz, rgb.contiguous())
        return centers.detach().cpu()


def visualize_attention_maps(
    attention_maps: Dict[str, torch.Tensor],
    layer_indices: List[int] = None,
    head_indices: List[int] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 12),
    cmap: str = 'viridis'
):
    """
    Visualize attention maps as heatmaps.

    Args:
        attention_maps: Dictionary from AttentionExtractor.extract()
        layer_indices: Which layers to plot (None for all)
        head_indices: Which heads to plot (None for all)
        save_path: Path to save figure (None to display)
        figsize: Figure size
        cmap: Colormap for heatmaps
    """
    # Get available layers
    available_layers = sorted([int(k.split('_')[1]) for k in attention_maps.keys()])

    if layer_indices is None:
        # Show first, middle, and last layers
        if len(available_layers) >= 3:
            layer_indices = [available_layers[0],
                           available_layers[len(available_layers)//2],
                           available_layers[-1]]
        else:
            layer_indices = available_layers

    # Get number of heads from first attention map
    first_key = f'layer_{available_layers[0]}'
    num_heads = attention_maps[first_key].shape[1]

    if head_indices is None:
        # Show 4 evenly spaced heads
        head_indices = [0, num_heads//4, num_heads//2, -1]
        head_indices = [h % num_heads for h in head_indices]

    # Create figure
    n_layers = len(layer_indices)
    n_heads = len(head_indices)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=figsize)
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if n_heads == 1:
        axes = axes.reshape(-1, 1)

    for i, layer_idx in enumerate(layer_indices):
        key = f'layer_{layer_idx}'
        if key not in attention_maps:
            continue

        attn = attention_maps[key][0]  # Take first batch item

        for j, head_idx in enumerate(head_indices):
            ax = axes[i, j]

            # Get attention matrix for this head
            attn_head = attn[head_idx].numpy()  # [seq_len, seq_len]

            # Plot heatmap
            im = ax.imshow(attn_head, cmap=cmap, aspect='auto')

            # Labels
            if i == 0:
                ax.set_title(f'Head {head_idx}', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'Layer {layer_idx}', fontsize=12)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Mark CLS token
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle('Attention Maps by Layer and Head\n(Red lines mark CLS token)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved attention heatmaps to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_attention_averaged_over_heads(
    attention_maps: Dict[str, torch.Tensor],
    layer_indices: List[int] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (18, 10),
    cmap: str = 'viridis'
):
    """
    Visualize attention maps AVERAGED over all heads for each layer.

    This provides a cleaner view of attention patterns without head-specific variations.

    Args:
        attention_maps: Dictionary from AttentionExtractor.extract()
        layer_indices: Which layers to plot (None for evenly spaced selection)
        save_path: Path to save figure (None to display)
        figsize: Figure size
        cmap: Colormap for heatmaps
    """
    # Get available layers
    available_layers = sorted([int(k.split('_')[1]) for k in attention_maps.keys()])

    if layer_indices is None:
        # Select ~6 evenly spaced layers
        n_select = min(6, len(available_layers))
        indices = np.linspace(0, len(available_layers) - 1, n_select, dtype=int)
        layer_indices = [available_layers[i] for i in indices]

    n_layers = len(layer_indices)

    # Create figure with 2 rows: full attention matrix and CLS attention bar
    fig, axes = plt.subplots(2, n_layers, figsize=figsize,
                             gridspec_kw={'height_ratios': [3, 1]})

    if n_layers == 1:
        axes = axes.reshape(-1, 1)

    for col, layer_idx in enumerate(layer_indices):
        key = f'layer_{layer_idx}'
        if key not in attention_maps:
            continue

        attn = attention_maps[key][0]  # [num_heads, seq_len, seq_len]

        # Average over heads
        attn_avg = attn.mean(dim=0).numpy()  # [seq_len, seq_len]

        # Top row: Full attention matrix
        ax_matrix = axes[0, col]
        im = ax_matrix.imshow(attn_avg, cmap=cmap, aspect='auto')
        ax_matrix.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')

        if col == 0:
            ax_matrix.set_ylabel('Query Token', fontsize=10)
        ax_matrix.set_xlabel('Key Token', fontsize=10)

        # Mark CLS token
        ax_matrix.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax_matrix.axvline(x=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)

        # Add colorbar
        plt.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

        # Bottom row: CLS attention to other tokens (bar chart)
        ax_bar = axes[1, col]
        cls_attention = attn_avg[0, 1:]  # CLS row, exclude self-attention

        # Create bar plot (subsample if too many tokens)
        n_tokens = len(cls_attention)
        if n_tokens > 50:
            # Subsample for visualization
            step = n_tokens // 50
            x_pos = np.arange(0, n_tokens, step)
            heights = cls_attention[::step]
        else:
            x_pos = np.arange(n_tokens)
            heights = cls_attention

        colors = plt.cm.get_cmap(cmap)(heights / (heights.max() + 1e-8))
        ax_bar.bar(x_pos, heights, color=colors, width=max(1, n_tokens//50))
        ax_bar.set_xlabel('Token Index', fontsize=10)
        if col == 0:
            ax_bar.set_ylabel('CLS Attention', fontsize=10)
        ax_bar.set_xlim(-1, n_tokens)

        # Add statistics as text
        ax_bar.text(0.95, 0.95, f'max: {cls_attention.max():.3f}\nentropy: {-np.sum(cls_attention * np.log(cls_attention + 1e-10)):.2f}',
                   transform=ax_bar.transAxes, fontsize=8, verticalalignment='top',
                   horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Attention Averaged Over All Heads\n(Top: Full matrix, Bottom: CLS→Tokens attention)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved head-averaged attention to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_cls_attention_across_layers(
    attention_maps: Dict[str, torch.Tensor],
    group_centers: np.ndarray = None,
    point_cloud: np.ndarray = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Visualize how CLS attention (averaged over heads) evolves across ALL layers.

    Creates a comprehensive visualization showing:
    1. Heatmap of CLS attention across layers (x-axis: tokens, y-axis: layers)
    2. Line plots of attention statistics across layers
    3. Optional: 3D visualization for selected layers

    Args:
        attention_maps: Dictionary from AttentionExtractor.extract()
        group_centers: Optional [num_groups, 3] for 3D subplot
        point_cloud: Optional [N, 3] for 3D subplot
        save_path: Path to save figure
        figsize: Figure size
    """
    # Get all layers sorted
    layer_indices = sorted([int(k.split('_')[1]) for k in attention_maps.keys()])
    n_layers = len(layer_indices)

    # Extract CLS attention for each layer (averaged over heads)
    cls_attention_matrix = []
    attention_stats = {
        'mean': [],
        'max': [],
        'entropy': [],
        'sparsity': []  # Top-10% concentration
    }

    for layer_idx in layer_indices:
        key = f'layer_{layer_idx}'
        attn = attention_maps[key][0]  # [heads, seq, seq]

        # Average over heads, get CLS row (excluding self)
        cls_attn = attn.mean(dim=0)[0, 1:].numpy()  # [num_tokens]
        cls_attention_matrix.append(cls_attn)

        # Compute statistics
        attention_stats['mean'].append(cls_attn.mean())
        attention_stats['max'].append(cls_attn.max())
        attention_stats['entropy'].append(-np.sum(cls_attn * np.log(cls_attn + 1e-10)))
        k = max(1, len(cls_attn) // 10)
        attention_stats['sparsity'].append(np.sort(cls_attn)[-k:].sum())

    cls_attention_matrix = np.array(cls_attention_matrix)  # [n_layers, n_tokens]

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Layout: Main heatmap on left, statistics on right
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])

    # 1. Main heatmap: CLS attention across layers
    ax_heatmap = fig.add_subplot(gs[0, 0])
    im = ax_heatmap.imshow(cls_attention_matrix, aspect='auto', cmap='viridis',
                           extent=[0, cls_attention_matrix.shape[1], n_layers-0.5, -0.5])
    ax_heatmap.set_xlabel('Token Index (Point Groups)', fontsize=11)
    ax_heatmap.set_ylabel('Layer', fontsize=11)
    ax_heatmap.set_title('CLS Attention Evolution Across Layers\n(Averaged over heads)', fontsize=12, fontweight='bold')
    ax_heatmap.set_yticks(range(n_layers))
    ax_heatmap.set_yticklabels(layer_indices)
    plt.colorbar(im, ax=ax_heatmap, label='Attention Weight')

    # 2. Statistics plots
    ax_stats = fig.add_subplot(gs[0, 1])

    ax_stats.plot(layer_indices, attention_stats['mean'], 'o-', label='Mean', color='blue')
    ax_stats.plot(layer_indices, attention_stats['max'], 's-', label='Max', color='red')
    ax_stats.plot(layer_indices, attention_stats['sparsity'], '^-', label='Top-10%', color='green')
    ax_stats.set_xlabel('Layer', fontsize=11)
    ax_stats.set_ylabel('Attention Value', fontsize=11)
    ax_stats.set_title('Attention Statistics', fontsize=12, fontweight='bold')
    ax_stats.legend(loc='best', fontsize=9)
    ax_stats.grid(True, alpha=0.3)

    # 3. Entropy plot
    ax_entropy = fig.add_subplot(gs[1, 1])
    ax_entropy.plot(layer_indices, attention_stats['entropy'], 'd-', color='purple', linewidth=2)
    ax_entropy.fill_between(layer_indices, attention_stats['entropy'], alpha=0.3, color='purple')
    ax_entropy.set_xlabel('Layer', fontsize=11)
    ax_entropy.set_ylabel('Entropy', fontsize=11)
    ax_entropy.set_title('Attention Entropy\n(Higher = More Uniform)', fontsize=11)
    ax_entropy.grid(True, alpha=0.3)

    # 4. Attention difference between consecutive layers
    ax_diff = fig.add_subplot(gs[1, 0])

    # Compute cosine similarity between consecutive layer attention patterns
    similarities = []
    for i in range(len(cls_attention_matrix) - 1):
        a = cls_attention_matrix[i]
        b = cls_attention_matrix[i + 1]
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        similarities.append(sim)

    ax_diff.bar(layer_indices[1:], similarities, color='teal', alpha=0.7)
    ax_diff.axhline(y=np.mean(similarities), color='red', linestyle='--', label=f'Mean: {np.mean(similarities):.3f}')
    ax_diff.set_xlabel('Layer', fontsize=11)
    ax_diff.set_ylabel('Cosine Similarity', fontsize=11)
    ax_diff.set_title('Attention Similarity Between Consecutive Layers', fontsize=11)
    ax_diff.legend(loc='best', fontsize=9)
    ax_diff.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved CLS attention evolution to {save_path}")
    else:
        plt.show()

    plt.close()

    return cls_attention_matrix, attention_stats


def visualize_layer_attention_on_pointcloud_grid(
    attention_maps: Dict[str, torch.Tensor],
    point_cloud: np.ndarray,
    group_centers: np.ndarray,
    layer_indices: List[int] = None,
    save_path: str = None
):
    """
    Visualize attention (averaged over heads) on point cloud for multiple layers in a grid.

    Creates an interactive HTML with subplots showing attention evolution.

    Args:
        attention_maps: Dictionary from AttentionExtractor.extract()
        point_cloud: Original point cloud [N, 3]
        group_centers: Center of each group [num_groups, 3]
        layer_indices: Which layers to show (None for auto-selection)
        save_path: Path to save HTML file
    """
    if not PLOTLY_AVAILABLE:
        logging.warning("Plotly not available. Skipping 3D grid visualization.")
        return

    available_layers = sorted([int(k.split('_')[1]) for k in attention_maps.keys()])

    if layer_indices is None:
        # Select 6 evenly spaced layers
        n_select = min(6, len(available_layers))
        indices = np.linspace(0, len(available_layers) - 1, n_select, dtype=int)
        layer_indices = [available_layers[i] for i in indices]

    n_layers = len(layer_indices)
    cols = min(3, n_layers)
    rows = (n_layers + cols - 1) // cols

    # Create subplot titles
    subplot_titles = [f'Layer {l}' for l in layer_indices]

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.08
    )

    for idx, layer_idx in enumerate(layer_indices):
        row = idx // cols + 1
        col = idx % cols + 1

        key = f'layer_{layer_idx}'
        attn = attention_maps[key][0]  # [heads, seq, seq]

        # Average over heads, get CLS attention
        cls_attn = attn.mean(dim=0)[0, 1:].numpy()  # [num_tokens]
        attn_norm = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)

        # Add point cloud (light gray background)
        fig.add_trace(
            go.Scatter3d(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                z=point_cloud[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.15),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )

        # Add attention-colored group centers
        fig.add_trace(
            go.Scatter3d(
                x=group_centers[:, 0],
                y=group_centers[:, 1],
                z=group_centers[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=attn_norm,
                    colorscale='Viridis',
                    opacity=0.9,
                    showscale=(idx == 0),  # Only show colorbar for first subplot
                    colorbar=dict(
                        title='Attention',
                        x=1.02,
                        len=0.5
                    ) if idx == 0 else None
                ),
                showlegend=False,
                text=[f'Token {i}<br>Attention: {cls_attn[i]:.4f}' for i in range(len(cls_attn))],
                hoverinfo='text'
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=dict(
            text='CLS Attention on Point Cloud (Averaged Over Heads) - Layer Comparison',
            font=dict(size=16)
        ),
        height=400 * rows,
        width=450 * cols,
        showlegend=False
    )

    # Update all scenes for consistent view
    for i in range(n_layers):
        scene_name = f'scene{i+1}' if i > 0 else 'scene'
        fig.update_layout(**{
            scene_name: dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                aspectmode='data'
            )
        })

    if save_path:
        fig.write_html(save_path)
        logging.info(f"Saved layer attention grid to {save_path}")
    else:
        fig.show()


def visualize_attention_on_pointcloud(
    point_cloud: np.ndarray,
    attention_weights: np.ndarray,
    group_centers: np.ndarray,
    title: str = "Attention Visualization",
    save_path: str = None,
    point_size: int = 2,
    center_size: int = 10
):
    """
    Visualize attention weights on the 3D point cloud.

    Args:
        point_cloud: Original point cloud [N, 3]
        attention_weights: Attention weights for each group [num_groups]
        group_centers: Center of each group [num_groups, 3]
        title: Plot title
        save_path: Path to save (None to display)
        point_size: Size of point cloud points
        center_size: Size of group center markers
    """
    if not PLOTLY_AVAILABLE:
        logging.warning("Plotly not available. Skipping 3D visualization.")
        return

    # Normalize attention weights for coloring
    attn_normalized = (attention_weights - attention_weights.min()) / \
                      (attention_weights.max() - attention_weights.min() + 1e-8)

    # Create figure
    fig = go.Figure()

    # Add original point cloud (gray)
    fig.add_trace(go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color='lightgray',
            opacity=0.3
        ),
        name='Point Cloud'
    ))

    # Add group centers colored by attention
    fig.add_trace(go.Scatter3d(
        x=group_centers[:, 0],
        y=group_centers[:, 1],
        z=group_centers[:, 2],
        mode='markers',
        marker=dict(
            size=center_size,
            color=attn_normalized,
            colorscale='Viridis',
            colorbar=dict(title='Attention'),
            opacity=0.9
        ),
        name='Group Centers (Attention)',
        text=[f'Attention: {w:.3f}' for w in attention_weights],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        width=900,
        height=700
    )

    if save_path:
        fig.write_html(save_path)
        logging.info(f"Saved 3D attention visualization to {save_path}")
    else:
        fig.show()


def visualize_attention_heads_on_pointcloud(
    point_cloud: np.ndarray,
    attention_weights: np.ndarray,
    group_centers: np.ndarray,
    head_indices: List[int] = None,
    title: str = "Attention by Head",
    save_path: str = None
):
    """
    Visualize attention from multiple heads on the point cloud.

    Args:
        point_cloud: Original point cloud [N, 3]
        attention_weights: Attention weights [num_heads, num_groups]
        group_centers: Center of each group [num_groups, 3]
        head_indices: Which heads to visualize
        title: Plot title
        save_path: Path to save
    """
    if not PLOTLY_AVAILABLE:
        logging.warning("Plotly not available. Skipping 3D visualization.")
        return

    num_heads = attention_weights.shape[0]

    if head_indices is None:
        head_indices = list(range(min(4, num_heads)))

    n_plots = len(head_indices)
    cols = min(2, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[f'Head {h}' for h in head_indices]
    )

    for idx, head_idx in enumerate(head_indices):
        row = idx // cols + 1
        col = idx % cols + 1

        attn = attention_weights[head_idx]
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        # Add point cloud
        fig.add_trace(
            go.Scatter3d(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                z=point_cloud[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.2),
                showlegend=False
            ),
            row=row, col=col
        )

        # Add attention-colored centers
        fig.add_trace(
            go.Scatter3d(
                x=group_centers[:, 0],
                y=group_centers[:, 1],
                z=group_centers[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=attn_norm,
                    colorscale='Viridis',
                    opacity=0.9
                ),
                showlegend=False,
                text=[f'Attn: {w:.3f}' for w in attn],
                hoverinfo='text'
            ),
            row=row, col=col
        )

    fig.update_layout(
        title=title,
        height=400 * rows,
        width=500 * cols
    )

    if save_path:
        fig.write_html(save_path)
        logging.info(f"Saved multi-head attention visualization to {save_path}")
    else:
        fig.show()


def visualize_attention_evolution(
    attention_maps: Dict[str, torch.Tensor],
    group_centers: np.ndarray,
    point_cloud: np.ndarray,
    head_idx: int = 0,
    save_path: str = None
):
    """
    Visualize how attention evolves across layers.

    Args:
        attention_maps: Dictionary from AttentionExtractor.extract()
        group_centers: Center of each group [num_groups, 3]
        point_cloud: Original point cloud [N, 3]
        head_idx: Which head to track
        save_path: Path to save
    """
    if not PLOTLY_AVAILABLE:
        logging.warning("Plotly not available. Skipping visualization.")
        return

    # Get layer indices
    layer_indices = sorted([int(k.split('_')[1]) for k in attention_maps.keys()])

    # Sample a few layers
    if len(layer_indices) > 6:
        sample_layers = [layer_indices[i] for i in
                        [0, len(layer_indices)//4, len(layer_indices)//2,
                         3*len(layer_indices)//4, -1]]
        sample_layers = list(set(sample_layers))
        sample_layers.sort()
    else:
        sample_layers = layer_indices

    n_layers = len(sample_layers)

    fig = make_subplots(
        rows=1, cols=n_layers,
        specs=[[{'type': 'scatter3d'} for _ in range(n_layers)]],
        subplot_titles=[f'Layer {l}' for l in sample_layers]
    )

    for col, layer_idx in enumerate(sample_layers, 1):
        key = f'layer_{layer_idx}'
        attn = attention_maps[key][0, head_idx, 0, 1:].numpy()  # CLS attention to tokens
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        # Point cloud
        fig.add_trace(
            go.Scatter3d(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                z=point_cloud[:, 2],
                mode='markers',
                marker=dict(size=1, color='lightgray', opacity=0.2),
                showlegend=False
            ),
            row=1, col=col
        )

        # Attention centers
        fig.add_trace(
            go.Scatter3d(
                x=group_centers[:, 0],
                y=group_centers[:, 1],
                z=group_centers[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=attn_norm,
                    colorscale='Viridis',
                    opacity=0.9
                ),
                showlegend=False
            ),
            row=1, col=col
        )

    fig.update_layout(
        title=f'Attention Evolution Across Layers (Head {head_idx})',
        height=500,
        width=350 * n_layers
    )

    if save_path:
        fig.write_html(save_path)
        logging.info(f"Saved attention evolution to {save_path}")
    else:
        fig.show()


def plot_attention_statistics(
    attention_maps: Dict[str, torch.Tensor],
    save_path: str = None
):
    """
    Plot statistics about attention patterns.

    Args:
        attention_maps: Dictionary from AttentionExtractor.extract()
        save_path: Path to save figure
    """
    layer_indices = sorted([int(k.split('_')[1]) for k in attention_maps.keys()])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean attention to CLS across layers
    ax1 = axes[0, 0]
    cls_attn_means = []
    cls_attn_stds = []
    for layer_idx in layer_indices:
        attn = attention_maps[f'layer_{layer_idx}'][0]  # [heads, seq, seq]
        cls_attn = attn[:, 0, 1:].mean(dim=1)  # Mean over tokens, per head
        cls_attn_means.append(cls_attn.mean().item())
        cls_attn_stds.append(cls_attn.std().item())

    ax1.errorbar(layer_indices, cls_attn_means, yerr=cls_attn_stds,
                 marker='o', capsize=3)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Mean Attention to CLS')
    ax1.set_title('CLS Token Attention Across Layers')
    ax1.grid(True, alpha=0.3)

    # 2. Attention entropy (uniformity) across layers
    ax2 = axes[0, 1]
    entropies = []
    for layer_idx in layer_indices:
        attn = attention_maps[f'layer_{layer_idx}'][0]  # [heads, seq, seq]
        # Compute entropy of attention distribution
        attn_flat = attn[:, 0, 1:]  # CLS to others
        entropy = -(attn_flat * torch.log(attn_flat + 1e-10)).sum(dim=-1).mean()
        entropies.append(entropy.item())

    ax2.plot(layer_indices, entropies, marker='s', color='green')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Attention Entropy')
    ax2.set_title('Attention Entropy (Higher = More Uniform)')
    ax2.grid(True, alpha=0.3)

    # 3. Head specialization (variance across heads) per layer
    ax3 = axes[1, 0]
    head_variances = []
    for layer_idx in layer_indices:
        attn = attention_maps[f'layer_{layer_idx}'][0]  # [heads, seq, seq]
        cls_attn = attn[:, 0, 1:]  # [heads, num_tokens]
        # Variance across heads for each token
        variance = cls_attn.var(dim=0).mean().item()
        head_variances.append(variance)

    ax3.plot(layer_indices, head_variances, marker='^', color='red')
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Variance Across Heads')
    ax3.set_title('Head Specialization (Higher = More Diverse Heads)')
    ax3.grid(True, alpha=0.3)

    # 4. Attention sparsity (how concentrated)
    ax4 = axes[1, 1]
    sparsities = []
    for layer_idx in layer_indices:
        attn = attention_maps[f'layer_{layer_idx}'][0]  # [heads, seq, seq]
        cls_attn = attn[:, 0, 1:]  # [heads, num_tokens]
        # Top-k attention concentration (what fraction goes to top 10%)
        k = max(1, cls_attn.shape[-1] // 10)
        topk_sum = cls_attn.topk(k, dim=-1)[0].sum(dim=-1).mean()
        sparsities.append(topk_sum.item())

    ax4.plot(layer_indices, sparsities, marker='d', color='purple')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Top-10% Attention Mass')
    ax4.set_title('Attention Sparsity (Higher = More Focused)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Saved attention statistics to {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# Main function for standalone usage
# =============================================================================

def main():
    """Main function for command-line usage."""
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from utils.params import parse_args
    from utils.load_models import load_vlm_model

    parser = argparse.ArgumentParser(description='Extract and visualize Uni3D attention maps')
    parser.add_argument('--pc-path', type=str, required=True,
                        help='Path to point cloud file (.npy or .npz)')
    parser.add_argument('--output-dir', type=str, default='./attention_vis',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--layers', type=int, nargs='+', default=None,
                        help='Specific layers to visualize')
    parser.add_argument('--heads', type=int, nargs='+', default=None,
                        help='Specific heads to visualize')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model (using default uni3d settings)
    logging.info("Loading Uni3D model...")
    model_args = parse_args()
    model_args.device = args.device
    _, model = load_vlm_model(model_args)
    model.eval()

    # Load point cloud
    logging.info(f"Loading point cloud from {args.pc_path}")
    if args.pc_path.endswith('.npz'):
        data = np.load(args.pc_path)
        pc = data['points'] if 'points' in data else data[list(data.keys())[0]]
    else:
        pc = np.load(args.pc_path)

    pc_tensor = torch.from_numpy(pc).float()
    if pc_tensor.dim() == 2:
        pc_tensor = pc_tensor.unsqueeze(0)

    # Initialize extractor
    extractor = AttentionExtractor(model, device=args.device)

    # Extract attention maps
    logging.info("Extracting attention maps...")
    attention_maps = extractor.extract(pc_tensor)

    # Get group centers for visualization
    group_centers = extractor.get_group_centers(pc_tensor).numpy()[0]

    # Generate visualizations
    logging.info("Generating visualizations...")

    # 1. Heatmaps
    visualize_attention_maps(
        attention_maps,
        layer_indices=args.layers,
        head_indices=args.heads,
        save_path=os.path.join(args.output_dir, 'attention_heatmaps.png')
    )

    # 2. Statistics
    plot_attention_statistics(
        attention_maps,
        save_path=os.path.join(args.output_dir, 'attention_statistics.png')
    )

    # 3. 3D visualizations (if plotly available)
    if PLOTLY_AVAILABLE:
        pc_np = pc_tensor[0, :, :3].numpy() if pc_tensor.shape[-1] >= 3 else pc_tensor[0].numpy()

        # Last layer attention on point cloud
        last_layer = max([int(k.split('_')[1]) for k in attention_maps.keys()])
        attn_last = attention_maps[f'layer_{last_layer}'][0, :, 0, 1:].mean(dim=0).numpy()

        visualize_attention_on_pointcloud(
            pc_np, attn_last, group_centers,
            title=f'Attention (Layer {last_layer}, Mean over Heads)',
            save_path=os.path.join(args.output_dir, 'attention_3d.html')
        )

        # Multi-head visualization
        attn_heads = attention_maps[f'layer_{last_layer}'][0, :, 0, 1:].numpy()
        visualize_attention_heads_on_pointcloud(
            pc_np, attn_heads, group_centers,
            head_indices=[0, 4, 8, 12] if attn_heads.shape[0] > 12 else None,
            title='Attention by Head',
            save_path=os.path.join(args.output_dir, 'attention_heads.html')
        )

        # Evolution across layers
        visualize_attention_evolution(
            attention_maps, group_centers, pc_np,
            head_idx=0,
            save_path=os.path.join(args.output_dir, 'attention_evolution.html')
        )

    logging.info(f"Done! Visualizations saved to {args.output_dir}")

    # Cleanup
    extractor.remove_hooks()


if __name__ == '__main__':
    main()
