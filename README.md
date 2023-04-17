# Clipped Noise Softmax

A PyTorch module implementing a noisy version of the softmax function, adding noise during training. To use this, you have to add it after the Softmax layer. By default, the noise is applied only for the values that are close to 0.1 or 0.9, and that can be adjusted with the parameter alpha.

Creator: Daniel Wiczew

# Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClippedNoiseSoftmax(nn.Module):
    """A PyTorch module implementing a clipped noise version of the softmax function, adding noise during training,
       but only when the value is close to the 0.9 and 0.1, which where the gradients are close to 0 and prevents proper
       training.
       To use this, you have to add it after the Softmax layer. By default the noise is applied only for the values
       that are close to 0.1 or 0.9, and that can be adjusted with parameter `alpha`.
       Creator: Daniel Wiczew
    """

    def __init__(self, stddev=1.0, alpha=0.1, log=False):
        """
        Initialize the ClippedNoiseSoftmax module.

        Args:
        stddev (float, optional): The standard deviation of the noise. Default is 1.0.
        alpha (float, optional): The threshold to control the noise filter. Default is 0.1.
        log (bool, optional): Whether the input is in log space. Default is False.
        """
        super().__init__()  # Initialize the parent class (nn.Module)
        self.stddev = stddev  # Standard deviation of the noise
        self.alpha = alpha  # Threshold for noise filter
        self.log = log  # Whether the input is in log space
        self.eps = 1e-3  # Small constant to prevent division by zero

    def center_function(self, x, min_value, max_value):
        return torch.where((x > min_value) & (x < max_value), x, torch.zeros_like(x))

    def filter_noise(self, x, noise):
        """
        Apply a noise filter based on the input values (x) and the generated noise.

        Args:
        x (Tensor): Input tensor.
        noise (Tensor): Noise tensor with the same shape as x.

        Returns:
        Tensor: The filtered noise tensor.
        """
        upper_limit = (
            1 - self.alpha
        )  # Calculate the upper limit for filtering (e.g., 0.9)
        lower_limit = self.alpha  # Calculate the lower limit for filtering (e.g., 0.1)

        # Calculate the noise for values greater than the upper limit
        noise_one = noise * (-1) * F.hardtanh(x, upper_limit, 1.0) / upper_limit

        # Calculate the noise for values between 0 and the lower limit
        noise_zero = noise * self.center_function(x, 0.0, lower_limit) / lower_limit

        # Combine both noises and clamp the result between 0 and 1
        return torch.clamp(noise_one + noise_zero, min=0.0, max=1.0)

    def forward(self, x):
        """
        Forward pass of the ClippedNoiseSoftmax module.

        Args:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor with the same shape as x.
        """
        if self.training:  # Check if the module is in training mode
            if self.log:  # If input is in log space, convert to normal space
                x = torch.exp(x)

            noise = (
                torch.rand_like(x) * self.stddev
            )  # Generate random noise with the same shape as x
            noise = self.filter_noise(x, noise)  # Apply the noise filter
            output = x + noise  # Add the filtered noise to the input

            if self.log:  # If input was in log space, convert back to log space
                output = torch.log(output)

            return output
        else:  # If not in training mode, return the input unchanged
            return x
```


# Usage

To use the `ClippedNoiseSoftmax` module in your neural network, simply add it after the softmax layer. Here's an example:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        self.noisy_softmax = ClippedNoiseSoftmax(stddev=1.0, alpha=0.1, log=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        x = self.noisy_softmax(x)
        return x

model = MyModel()
```


# Installation

To install the `ClippedNoiseSoftmax` module, simply save the code in a Python file (e.g., `clipped_noise_softmax.py`) and import it into your project.


# Example

Here is an example of how to train a neural network with the ClippedNoiseSoftmax layer using the MNIST dataset:


```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from clipped_noisy_softmax import ClippedNoiseSoftmax

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        self.noisy_softmax = ClippedNoiseSoftmax(stddev=1.0, alpha=0.1, log=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        x = self.noisy_softmax(x)
        return x

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Initialize the model, loss function, and optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item()}")

# Test the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data.view(-1, 784))
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy: {correct / total * 100}%")
```

This example demonstrates how to train and test a neural network with the ClippedNoiseSoftmax layer on the MNIST dataset. The model learns with added noise during training, potentially improving its generalization capabilities.

# License

This project is licensed under the MIT License.

# References
Similar to that below, so I put a reference
Chen, Binghui, Weihong Deng, and Junping Du. "Noisy softmax: Improving the generalization ability of dcnn via postponing the early softmax saturation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
