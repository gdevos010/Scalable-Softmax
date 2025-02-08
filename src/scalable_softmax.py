import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScalableSoftmax(nn.Module):
    """Scalable-Softmax (SSMax) implementation from the paper 
    'Scalable-Softmax Is Superior for Attention'.
    
    This is a drop-in replacement for standard Softmax that helps prevent attention
    fading in transformers by incorporating input size scaling. The scaling helps maintain
    focused attention distributions even with large input sizes.
    
    Args:
        s (float, optional): Scaling parameter that controls attention focusing strength.
            Lower values (e.g. 0.1) produce sharper attention, higher values (e.g. 1.0)
            produce softer attention. Default: 0.43 as used in paper.
        learn_scaling (bool, optional): If True, make scaling parameter learnable.
            Default: True
        bias (bool, optional): If True, adds a learnable bias term. The paper found
            that while bias helps training, it can hurt length generalization.
            Default: False
    
    Shape:
        - Input: (*, N) where * is any number of dimensions and N is the sequence length
        - Output: Same shape as input
    """
    def __init__(self, s: float = 0.43, learn_scaling: bool = True, bias: bool = False):
        super().__init__()
        
        # Initialize scaling parameter
        if learn_scaling:
            self.s = nn.Parameter(torch.tensor(s, dtype=torch.float))
        else:
            self.register_buffer('s', torch.tensor(s, dtype=torch.float))
            
        # Optional bias parameter 
        if bias:
            self.b = nn.Parameter(torch.zeros(1))
        else:
            self.b = None
            
    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Forward pass applying SSMax along specified dimension.
        
        Args:
            x (torch.Tensor): Input tensor
            dim (int): Dimension along which to apply SSMax. Default: -1
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        # Get size of dimension we're applying SSMax to
        n = x.size(dim)
        
        # Apply scaling factor based on input size
        if self.b is not None:
            # Version with bias term
            x_scaled = (self.s * math.log(n) + self.b) * x
        else:
            # Standard version from paper
            x_scaled = self.s * math.log(n) * x
            
        # Apply standard softmax
        return F.softmax(x_scaled, dim=dim)

    def extra_repr(self) -> str:
        """String representation of module."""
        return f's={self.s.item():.3f}, bias={self.b is not None}'

