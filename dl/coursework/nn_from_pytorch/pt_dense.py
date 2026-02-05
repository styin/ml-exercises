import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()
        # Define the layer’s learnable parameters:
        # You can either create your own parameters…
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.ones(output_dim))
        
        # Alternatively, you can use nn.Linear as a building block:
        # self.linear = nn.Linear(input_dim, output_dim)
        # nn.init.normal_(self.linear.weight)
        # nn.init.constant_(self.linear.bias, 1.0)
    
    def forward(self, x):
        # If using custom parameters:
        z = torch.matmul(x, self.weight) + self.bias
        
        # If using nn.Linear instead:
        # z = self.linear(x)
        
        # Apply the sigmoid activation function:
        return torch.sigmoid(z)

if __name__ == "__main__":
    # Create an instance of the layer with 5 input features and 3 outputs.
    layer = DenseLayer(input_dim=5, output_dim=3)

    # Generate a dummy input tensor (e.g., a batch of 2 samples)
    x = torch.randn(2, 5)

    # Compute the forward pass
    output = layer(x)
    
    print(f'Forward pass: {output}')
    print(f'Layer weights: {layer.weight}') 
    print(f'Layer bias: {layer.bias}')