"""
Fraunhofer Diffraction Simulator

Implements differentiable Fraunhofer diffraction using PyTorch.
Uses FFT-based computation for efficient forward propagation.
"""

import torch
import torch.nn as nn
import numpy as np


class FraunhoferDiffraction(nn.Module):
    """
    Differentiable Fraunhofer diffraction simulator.
    
    Fraunhofer diffraction describes far-field diffraction patterns,
    where the diffraction pattern is the Fourier transform of the aperture.
    
    Args:
        wavelength (float): Wavelength of light (e.g., 193nm for DUV lithography)
        pixel_size (float): Physical size of each pixel in the mask (in nm)
        NA (float): Numerical aperture of the optical system
        use_coherent (bool): If True, use coherent illumination; otherwise partially coherent
    """
    
    def __init__(self, wavelength=13.5, pixel_size=1.0, NA=0.33, use_coherent=True):
        super(FraunhoferDiffraction, self).__init__()
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.NA = NA
        self.use_coherent = use_coherent
        
    def forward(self, mask):
        """
        Compute Fraunhofer diffraction pattern.
        
        Args:
            mask (torch.Tensor): Input mask pattern, shape (batch, 1, H, W) or (H, W)
                                Values typically in [0, 1] where 1 is transparent
        
        Returns:
            torch.Tensor: Intensity pattern at the image plane, same shape as input
        """
        # Ensure proper shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        batch_size, channels, height, width = mask.shape
        
        # Apply Fourier transform (represents far-field diffraction)
        mask_fft = torch.fft.fft2(mask)
        mask_fft_shifted = torch.fft.fftshift(mask_fft)
        
        # Apply pupil function (limited by numerical aperture)
        pupil = self._create_pupil_function(height, width, mask.device)
        filtered_fft = mask_fft_shifted * pupil
        
        # Inverse transform to get image plane
        filtered_fft_unshifted = torch.fft.ifftshift(filtered_fft)
        image_field = torch.fft.ifft2(filtered_fft_unshifted)
        
        # Calculate intensity (squared magnitude)
        intensity = torch.abs(image_field) ** 2
        
        return intensity
    
    def _create_pupil_function(self, height, width, device):
        """
        Create circular pupil function based on numerical aperture.
        
        The cutoff spatial frequency of the optical system is NA/λ (in
        cycles/nm).  ``torch.fft.fftfreq`` with ``d=pixel_size`` returns
        frequencies in cycles/nm, so the cutoff can be applied directly.
        
        Args:
            height (int): Height of the mask
            width (int): Width of the mask
            device: PyTorch device
            
        Returns:
            torch.Tensor: Pupil function (in shifted/centered coordinates)
        """
        # Create frequency grid in physical units (cycles / nm)
        fy = torch.fft.fftshift(torch.fft.fftfreq(height, d=self.pixel_size)).to(device)
        fx = torch.fft.fftshift(torch.fft.fftfreq(width, d=self.pixel_size)).to(device)
        
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
        
        # Frequency magnitude (cycles / nm)
        freq_mag = torch.sqrt(fx_grid**2 + fy_grid**2)
        
        # Physical cutoff: NA / wavelength (cycles / nm)
        cutoff = self.NA / self.wavelength
        
        # Circular pupil - allow frequencies up to cutoff
        pupil = (freq_mag <= cutoff).float()
        
        return pupil.unsqueeze(0).unsqueeze(0)
    
    def simulate_partial_coherence(self, mask, n_sources=5):
        """
        Simulate partially coherent illumination using multiple point sources.
        
        Args:
            mask (torch.Tensor): Input mask pattern
            n_sources (int): Number of point sources to simulate
            
        Returns:
            torch.Tensor: Intensity pattern with partial coherence
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        batch_size, channels, height, width = mask.shape
        
        # Create multiple source points in a circular pattern
        total_intensity = torch.zeros_like(mask, dtype=torch.float32)
        
        for i in range(n_sources):
            angle = 2 * np.pi * i / n_sources
            # Phase shift for off-axis illumination
            phase_shift = self._create_phase_shift(height, width, angle, mask.device)
            shifted_mask = mask * phase_shift
            
            # Compute coherent diffraction for this source
            intensity = self.forward(shifted_mask)
            total_intensity += intensity
            
        # Average intensity
        total_intensity = total_intensity / n_sources
        
        return total_intensity
    
    def _create_phase_shift(self, height, width, angle, device):
        """Create phase shift for off-axis illumination."""
        y = torch.arange(height, device=device) - height / 2
        x = torch.arange(width, device=device) - width / 2
        
        x_grid, y_grid = torch.meshgrid(x, y, indexing='xy')
        
        # Small offset for off-axis illumination
        offset = 0.1 * self.NA
        phase = offset * (x_grid * np.cos(angle) + y_grid * np.sin(angle))
        
        return torch.exp(1j * phase).unsqueeze(0).unsqueeze(0)


def create_test_mask(size=128, pattern_type='square'):
    """
    Create test mask patterns.
    
    Args:
        size (int): Size of the mask (square)
        pattern_type (str): Type of pattern ('square', 'circle', 'lines')
        
    Returns:
        torch.Tensor: Test mask pattern
    """
    mask = torch.zeros((size, size))
    center = size // 2
    
    if pattern_type == 'square':
        # Square aperture
        square_size = size // 4
        mask[center-square_size:center+square_size, 
             center-square_size:center+square_size] = 1.0
             
    elif pattern_type == 'circle':
        # Circular aperture
        y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
        radius = size // 4
        dist = torch.sqrt((x - center)**2 + (y - center)**2)
        mask[dist <= radius] = 1.0
        
    elif pattern_type == 'lines':
        # Grating pattern (lines and spaces)
        period = size // 8
        for i in range(0, size, period * 2):
            mask[:, i:i+period] = 1.0
            
    return mask
