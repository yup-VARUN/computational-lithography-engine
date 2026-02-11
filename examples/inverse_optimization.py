"""
Example: Inverse Mask Optimization

This example demonstrates inverse lithography technology (ILT) where
we optimize a mask to produce a desired target pattern.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litho_engine import FraunhoferDiffraction, MaskOptimizer
from litho_engine.optimizer import AdaptiveMaskOptimizer


def create_target_pattern(size=128, pattern_type='H_letter'):
    """Create target patterns to optimize for."""
    target = torch.zeros((size, size))
    center = size // 2
    
    if pattern_type == 'H_letter':
        # Create an 'H' shaped pattern
        bar_width = size // 10
        bar_height = size // 2
        
        # Left vertical bar
        target[center-bar_height:center+bar_height, 
               center-bar_height//2:center-bar_height//2+bar_width] = 1.0
        
        # Right vertical bar
        target[center-bar_height:center+bar_height, 
               center+bar_height//2-bar_width:center+bar_height//2] = 1.0
        
        # Horizontal bar
        target[center-bar_width//2:center+bar_width//2,
               center-bar_height//2:center+bar_height//2] = 1.0
               
    elif pattern_type == 'cross':
        # Create a cross pattern
        bar_width = size // 8
        bar_length = size // 2
        
        # Horizontal bar
        target[center-bar_width//2:center+bar_width//2, 
               center-bar_length:center+bar_length] = 1.0
        
        # Vertical bar
        target[center-bar_length:center+bar_length,
               center-bar_width//2:center+bar_width//2] = 1.0
               
    elif pattern_type == 'rings':
        # Create concentric rings
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        dist = torch.sqrt((x - center)**2 + (y - center)**2)
        
        for r in range(size//8, size//3, size//12):
            mask_ring = (dist >= r - size//20) & (dist <= r + size//20)
            target[mask_ring] = 1.0
            
    return target


def main():
    print("=" * 60)
    print("Inverse Mask Optimization (ILT)")
    print("=" * 60)
    
    # Setup diffraction model (EUV parameters)
    wavelength = 13.5   # nm (EUV)
    pixel_size = 1.0    # nm
    NA = 0.33
    
    diffraction = FraunhoferDiffraction(
        wavelength=wavelength,
        pixel_size=pixel_size,
        NA=NA,
        use_coherent=True
    )
    
    print(f"\nDiffraction Model Parameters:")
    print(f"  Wavelength: {wavelength} nm (EUV)")
    print(f"  Numerical Aperture: {NA}")
    
    # Create target pattern
    mask_size = 128
    target_pattern = create_target_pattern(size=mask_size, pattern_type='H_letter')
    
    print(f"\nTarget Pattern:")
    print(f"  Size: {mask_size}x{mask_size}")
    print(f"  Target coverage: {target_pattern.mean():.2%}")
    
    # Initialize optimizer
    print(f"\n--- Starting Optimization ---")
    optimizer = AdaptiveMaskOptimizer(
        diffraction_model=diffraction,
        mask_shape=(mask_size, mask_size),
        learning_rate=0.05,
        regularization=0.01,
        use_scheduler=True
    )
    
    # Run optimization
    num_iterations = 150
    results = optimizer.optimize(
        target=target_pattern,
        num_iterations=num_iterations,
        loss_type='mse',
        threshold=0.5,
        verbose=True,
        early_stopping_patience=30
    )
    
    # Get optimized mask and results
    optimized_mask = results['mask']
    loss_history = results['loss_history']
    
    print(f"\nOptimization Results:")
    print(f"  Final Loss: {results['final_loss']:.6f}")
    print(f"  Total Iterations: {len(loss_history)}")
    
    # Verify the optimized mask
    with torch.no_grad():
        predicted_pattern = diffraction(optimized_mask).squeeze()
    
    # Calculate metrics
    mse = torch.mean((predicted_pattern - target_pattern)**2).item()
    print(f"  Final MSE: {mse:.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Target pattern
    axes[0, 0].imshow(target_pattern.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Target Pattern', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Optimized mask
    axes[0, 1].imshow(optimized_mask.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Optimized Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Predicted pattern
    axes[1, 0].imshow(predicted_pattern.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Predicted Pattern', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Loss history
    axes[1, 1].plot(loss_history, linewidth=2)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('Optimization Progress', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    output_path = 'inverse_optimization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    print("\nOptimization complete!")


if __name__ == "__main__":
    main()
