"""
Inverse Mask Optimizer

Implements inverse lithography technology (ILT) using gradient-based optimization
to find the optimal mask pattern that produces a desired target pattern.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .diffraction import FraunhoferDiffraction


class MaskOptimizer:
    """
    Inverse mask optimizer for computational lithography.
    
    Uses gradient-based optimization to find the mask pattern that
    produces the desired target pattern when passed through the
    lithography system.
    
    Args:
        diffraction_model (FraunhoferDiffraction): The forward diffraction model
        mask_shape (tuple): Shape of the mask (height, width)
        learning_rate (float): Learning rate for optimization
        regularization (float): Regularization weight for mask smoothness
    """
    
    def __init__(self, diffraction_model, mask_shape, learning_rate=0.01, 
                 regularization=0.001):
        self.diffraction_model = diffraction_model
        self.mask_shape = mask_shape
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Initialize mask with random values
        self.mask = nn.Parameter(
            torch.rand(mask_shape, requires_grad=True)
        )
        
        # Optimizer
        self.optimizer = optim.Adam([self.mask], lr=learning_rate)
        
    def optimize(self, target, num_iterations=100, loss_type='mse', 
                 threshold=None, verbose=True):
        """
        Optimize mask to match target pattern.
        
        Args:
            target (torch.Tensor): Desired intensity pattern
            num_iterations (int): Number of optimization iterations
            loss_type (str): Type of loss function ('mse', 'l1', 'bce')
            threshold (float): Optional threshold for binary target
            verbose (bool): Print progress
            
        Returns:
            dict: Optimization results including final mask and loss history
        """
        target = target.float()
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 3:
            target = target.unsqueeze(1)
            
        loss_history = []
        
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            
            # Forward pass through diffraction model
            # Apply sigmoid to constrain mask values to [0, 1]
            mask_constrained = torch.sigmoid(self.mask)
            predicted = self.diffraction_model(mask_constrained)
            
            # Compute loss
            loss = self._compute_loss(predicted, target, loss_type)
            
            # Add regularization (encourages smooth masks)
            if self.regularization > 0:
                reg_loss = self._regularization_loss(mask_constrained)
                total_loss = loss + self.regularization * reg_loss
            else:
                total_loss = loss
                
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            loss_history.append(total_loss.item())
            
            if verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
                print(f"Iteration {iteration}: Loss = {total_loss.item():.6f}")
                
        # Get final optimized mask
        final_mask = torch.sigmoid(self.mask).detach()
        
        # Apply threshold if specified
        if threshold is not None:
            final_mask = (final_mask > threshold).float()
            
        return {
            'mask': final_mask,
            'loss_history': loss_history,
            'final_loss': loss_history[-1]
        }
    
    def _compute_loss(self, predicted, target, loss_type='mse'):
        """
        Compute loss between predicted and target patterns.
        
        Args:
            predicted (torch.Tensor): Predicted intensity pattern
            target (torch.Tensor): Target intensity pattern
            loss_type (str): Type of loss function
            
        Returns:
            torch.Tensor: Loss value
        """
        if loss_type == 'mse':
            # Mean squared error
            loss = nn.functional.mse_loss(predicted, target)
        elif loss_type == 'l1':
            # L1 loss (mean absolute error)
            loss = nn.functional.l1_loss(predicted, target)
        elif loss_type == 'bce':
            # Binary cross entropy (for binary patterns)
            loss = nn.functional.binary_cross_entropy(predicted, target)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        return loss
    
    def _regularization_loss(self, mask):
        """
        Compute regularization loss to encourage smooth masks.
        Uses total variation (TV) regularization.
        
        Args:
            mask (torch.Tensor): Current mask
            
        Returns:
            torch.Tensor: Regularization loss
        """
        # Total variation regularization
        # Penalizes large gradients (encourages smoothness)
        diff_h = torch.abs(mask[1:, :] - mask[:-1, :])
        diff_v = torch.abs(mask[:, 1:] - mask[:, :-1])
        
        tv_loss = torch.mean(diff_h) + torch.mean(diff_v)
        
        return tv_loss
    
    def set_initial_mask(self, initial_mask):
        """
        Set initial mask for optimization.
        
        Args:
            initial_mask (torch.Tensor): Initial mask pattern
        """
        # Use inverse sigmoid to map [0,1] to unbounded space
        eps = 1e-7
        initial_mask_clamped = torch.clamp(initial_mask, eps, 1 - eps)
        self.mask.data = torch.logit(initial_mask_clamped)
        
    def get_current_mask(self):
        """
        Get current mask in [0, 1] range.
        
        Returns:
            torch.Tensor: Current mask
        """
        return torch.sigmoid(self.mask).detach()


class AdaptiveMaskOptimizer(MaskOptimizer):
    """
    Enhanced mask optimizer with adaptive learning rate and advanced features.
    """
    
    def __init__(self, diffraction_model, mask_shape, learning_rate=0.01,
                 regularization=0.001, use_scheduler=True):
        super().__init__(diffraction_model, mask_shape, learning_rate, regularization)
        
        # Use learning rate scheduler
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=20
            )
        else:
            self.scheduler = None
            
    def optimize(self, target, num_iterations=100, loss_type='mse',
                 threshold=None, verbose=True, early_stopping_patience=30):
        """
        Optimize with adaptive learning rate and early stopping.
        """
        target = target.float()
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 3:
            target = target.unsqueeze(1)
            
        loss_history = []
        best_loss = float('inf')
        patience_counter = 0
        best_mask = None
        
        for iteration in range(num_iterations):
            self.optimizer.zero_grad()
            
            # Forward pass
            mask_constrained = torch.sigmoid(self.mask)
            predicted = self.diffraction_model(mask_constrained)
            
            # Compute loss
            loss = self._compute_loss(predicted, target, loss_type)
            
            if self.regularization > 0:
                reg_loss = self._regularization_loss(mask_constrained)
                total_loss = loss + self.regularization * reg_loss
            else:
                total_loss = loss
                
            # Backward pass
            total_loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(total_loss)
                
            loss_history.append(total_loss.item())
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_mask = mask_constrained.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping at iteration {iteration}")
                break
                
            if verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Iteration {iteration}: Loss = {total_loss.item():.6f}, LR = {current_lr:.6f}")
                
        # Use best mask
        final_mask = best_mask if best_mask is not None else torch.sigmoid(self.mask).detach()
        
        if threshold is not None:
            final_mask = (final_mask > threshold).float()
            
        return {
            'mask': final_mask,
            'loss_history': loss_history,
            'final_loss': best_loss
        }
