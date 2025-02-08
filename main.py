import torch
from scalable_softmax import ScalableSoftmax

# Initialize with default parameters
smax = ScalableSoftmax()

# Or customize parameters
smax = ScalableSoftmax(
    s=0.43,  # scaling parameter
    learn_scaling=True,  # make scaling parameter learnable
    bias=False  # whether to use bias term
)

# Apply to input tensor
batch_size = 32
sequence_length = 128
x = torch.randn(batch_size, sequence_length)
output = smax(x)