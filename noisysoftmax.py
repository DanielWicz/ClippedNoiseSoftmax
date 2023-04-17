class NoisySoftmax(nn.Module):
     """
   A PyTorch module implementing a noisy version of the softmax function, adding noise during training.
   To use this, you have to add it after the Softmax layer. By default the noise is applied only for the values
   that are close to 0.1 or 0.9, and that can be adjusted with parameter `alpha`.
   Creator: Daniel Wiczew
     """

   def __init__(self, stddev=1.0, alpha=0.1, log=False):
       """
       Initialize the NoisySoftmax module.

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

   def filter_noise(self, x, noise):
       """
       Apply a noise filter based on the input values (x) and the generated noise.

       Args:
       x (Tensor): Input tensor.
       noise (Tensor): Noise tensor with the same shape as x.

       Returns:
       Tensor: The filtered noise tensor.
       """
       upper_limit = 1 - self.alpha  # Calculate the upper limit for filtering (e.g., 0.9)
       lower_limit = self.alpha  # Calculate the lower limit for filtering (e.g., 0.1)
       
       # Calculate the noise for values greater than the upper limit
       noise_one = noise * (-1) * F.hardtanh(x, upper_limit, 1.0) / upper_limit
       
       # Calculate the noise for values between 0 and the lower limit
       noise_zero = noise * center_function(x, 0.0, lower_limit) / lower_limit
       
       # Combine both noises and clamp the result between 0 and 1
       return torch.clamp(noise_one + noise_zero, min=0.0, max=1.0)

   def forward(self, x):
       """
       Forward pass of the NoisySoftmax module.

       Args:
       x (Tensor): Input tensor.

       Returns:
       Tensor: Output tensor with the same shape as x.
       """
       if self.training:  # Check if the module is in training mode
           if self.log:  # If input is in log space, convert to normal space
               x = torch.exp(x)
               
           noise = torch.rand_like(x) * self.stddev  # Generate random noise with the same shape as x
           noise = self.filter_noise(x, noise)  # Apply the noise filter
           output = x + noise  # Add the filtered noise to the input
           
           if self.log:  # If input was in log space, convert back to log space
               output = torch.log(output)
               
           return output
       else:  # If not in training mode, return the input unchanged
           return x
