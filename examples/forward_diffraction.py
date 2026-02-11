"""
Example: Forward Fraunhofer Diffraction Simulation

This example demonstrates the forward simulation of Fraunhofer diffraction
for various mask patterns.
"""

import torch
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from litho_engine import FraunhoferDiffraction
from litho_engine.diffraction import create_test_mask


def main():
    print("=" * 60)
    print("Forward Fraunhofer Diffraction Simulation")
    print("=" * 60)
    
    # Create diffraction model
    # Parameters for EUV lithography at 13.5nm
    wavelength = 13.5   # nm (EUV)
    pixel_size = 1.0    # nm
    NA = 0.33           # Numerical aperture
    
    diffraction = FraunhoferDiffraction(
        wavelength=wavelength,
        pixel_size=pixel_size,
        NA=NA,
        use_coherent=True
    )
    
    print(f"\nDiffraction Model Parameters:")
    print(f"  Wavelength: {wavelength} nm")
    print(f"  Pixel Size: {pixel_size} nm")
    print(f"  Numerical Aperture: {NA}")
    
    # Test different mask patterns
    patterns = ['square', 'circle', 'lines']
    mask_size = 128
    
    fig, axes = plt.subplots(len(patterns), 2, figsize=(10, 12))
    
    for idx, pattern_type in enumerate(patterns):
        print(f"\n--- Testing {pattern_type.upper()} pattern ---")
        
        # Create test mask
        mask = create_test_mask(size=mask_size, pattern_type=pattern_type)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask value range: [{mask.min():.2f}, {mask.max():.2f}]")
        
        # Simulate diffraction
        with torch.no_grad():
            intensity = diffraction(mask)
            
        # Remove batch dimensions for visualization
        intensity = intensity.squeeze()
        
        print(f"Intensity shape: {intensity.shape}")
        print(f"Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")
        
        # Plot mask and intensity
        axes[idx, 0].imshow(mask.cpu().numpy(), cmap='gray')
        axes[idx, 0].set_title(f'{pattern_type.capitalize()} Mask')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(intensity.cpu().numpy(), cmap='hot')
        axes[idx, 1].set_title(f'Diffraction Pattern')
        axes[idx, 1].axis('off')
    
    plt.tight_layout()
    output_path = 'diffraction_patterns.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
