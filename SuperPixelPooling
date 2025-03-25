import torch
import torch.nn.functional as F

class GridSuperPixelPooling(torch.nn.Module):
    """
    Approximates Grid Super-Pixel pooling using a regular grid in pure Python with PyTorch.
    
    Args:
        S (int): Number of grid cells along each spatial dimension.
        pool_type (str): Type of pooling ('max' or 'mean').
    """
    def __init__(self, S, pool_type='max'):
        super(GridSuperPixelPooling, self).__init__()
        self.S = S
        self.pool_type = pool_type

    def forward(self, x):
        """
        Forward pass for grid-based superpixel pooling.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Pooled tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        k_h = H // self.S  # Height of each grid cell
        k_w = W // self.S  # Width of each grid cell
        
        # Apply pooling
        if self.pool_type == 'max':
            pooled = F.max_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w))
        elif self.pool_type == 'mean':
            pooled = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w))
        else:
            raise ValueError("pool_type must be 'max' or 'mean'")
        
        # Upsample back to original size
        output = F.interpolate(pooled, size=(H, W), mode='nearest')
        return output
