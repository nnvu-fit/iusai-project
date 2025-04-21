import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Subset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import dataset as ds

class SimpleSinusoidalPositionalEmbedding(nn.Module):
  """Simple sinusoidal positional embedding for time steps."""
  def __init__(self, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim
    
  def forward(self, timesteps):
    half_dim = self.embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if self.embedding_dim % 2 == 1:  # if odd dimension, pad with zeros
      emb = F.pad(emb, (0, 1, 0, 0))
    return emb

class UNet(nn.Module):
  """Simple U-Net architecture for diffusion models."""
  def __init__(self, in_channels, hidden_dim=64, time_embedding_dim=128):
    super().__init__()
    self.time_embedding = SimpleSinusoidalPositionalEmbedding(time_embedding_dim)
    
    # Initial convolution
    self.init_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
    
    # Downsampling path
    self.down1 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=4, stride=2, padding=1)
    self.down2 = nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1)
    
    # Time embedding projection
    self.time_mlp = nn.Sequential(
      nn.Linear(time_embedding_dim, hidden_dim*4),
      nn.SiLU(),
      nn.Linear(hidden_dim*4, hidden_dim*4)
    )
    
    # Middle blocks
    self.mid_block1 = nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1)
    self.mid_block2 = nn.Conv2d(hidden_dim*4, hidden_dim*4, kernel_size=3, padding=1)
    
    # Upsampling path
    self.up1 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=2, padding=1)
    self.up2 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=4, stride=2, padding=1)
    
    # Final convolution
    self.final_conv = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
    
  def forward(self, x, timestep):
    # Embed time
    t_emb = self.time_embedding(timestep)
    t_emb = self.time_mlp(t_emb)
    
    # Initial features
    h = F.silu(self.init_conv(x))
    
    # Downsample
    h1 = F.silu(self.down1(h))
    h2 = F.silu(self.down2(h1))
    
    # Add time embedding
    h2 = h2 + t_emb[:, :, None, None]
    
    # Middle
    h2 = F.silu(self.mid_block1(h2))
    h2 = F.silu(self.mid_block2(h2))
    
    # Upsample
    h = F.silu(self.up1(h2))
    h = F.silu(self.up2(h))
    
    # Final output (predict noise)
    return self.final_conv(h)

class DiffusionModel:
  def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Initialize a diffusion model.
    
    Args:
      model: The noise prediction network
      beta_start: Start value for noise schedule
      beta_end: End value for noise schedule
      timesteps: Number of diffusion steps
      device: Device to run the model on
    """
    self.model = model.to(device)
    self.device = device
    self.timesteps = timesteps
    
    # Define noise schedule
    self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
    self.alphas = 1 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    # Pre-compute values for diffusion process
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
    self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)
    self.posterior_variance = self.betas * (1 - self.alphas_cumprod.roll(1)) / (1 - self.alphas_cumprod)
    self.posterior_variance[0] = self.betas[0]
    
  def diffusion_forward(self, x_0, t):
    """Forward diffusion process: q(x_t | x_0)"""
    noise = torch.randn_like(x_0)
    mean = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_0
    var = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return mean + var * noise, noise
  
  def p_sample(self, x_t, t):
    """Single sampling step: p(x_{t-1} | x_t)"""
    with torch.no_grad():
      # Get model's predicted noise
      predicted_noise = self.model(x_t, t)
      
      # Calculate mean
      alpha = self.alphas[t].view(-1, 1, 1, 1)
      alpha_cumprod = self.alphas_cumprod[t].view(-1, 1, 1, 1)
      beta = self.betas[t].view(-1, 1, 1, 1)
      
      # Compute the mean of p(x_{t-1} | x_t)
      pred_x0 = (x_t - beta * predicted_noise / torch.sqrt(1 - alpha_cumprod)) / torch.sqrt(alpha)
      mean = pred_x0
      
      # Add variance
      noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
      variance = self.posterior_variance[t].view(-1, 1, 1, 1) * noise
      
      return mean + variance
  
  def sample(self, shape, steps=None):
    """Sample images from noise using the diffusion model"""
    steps = steps or self.timesteps
    
    # Start with pure noise
    x = torch.randn(shape).to(self.device)
    
    # Progressively denoise
    for t in tqdm(range(self.timesteps-1, -1, -1)):
      # Expand t to batch dimension
      t_batch = torch.tensor([t] * shape[0], device=self.device)
      x = self.p_sample(x, t_batch)
      
    # Rescale to [0, 1]
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    
    return x
  
  def train_step(self, optimizer, x_0, loss_fn=F.mse_loss):
    """Single training step for the diffusion model"""
    optimizer.zero_grad()
    
    # Sample timesteps
    t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=self.device)
    
    # Forward diffusion process
    x_t, noise = self.diffusion_forward(x_0, t)
    
    # Predict noise
    predicted_noise = self.model(x_t, t)
    
    # Compute loss
    loss = loss_fn(predicted_noise, noise)
    
    # Backprop
    loss.backward()
    optimizer.step()
    
    return loss.item()
  
  def evaluate_dataset(self, dataloader, num_epochs=5, lr=2e-4, save_samples=True, sample_every=5):
    """Train on a dataset and evaluate it by generating samples"""
    optimizer = optim.Adam(self.model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(num_epochs):
      epoch_losses = []
      pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
      
      for batch in pbar:
        if isinstance(batch, list) or isinstance(batch, tuple):
          x = batch[0].to(self.device)
        else:
          x = batch.to(self.device)
        
        # Scale images to [-1, 1]
        if x.min() >= 0 and x.max() <= 1:
          x = 2 * x - 1
        
        loss = self.train_step(optimizer, x)
        epoch_losses.append(loss)
        pbar.set_postfix(loss=f"{loss:.4f}")
      
      avg_loss = sum(epoch_losses) / len(epoch_losses)
      losses.append(avg_loss)
      print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
      
      # Generate and save samples
      if save_samples and (epoch + 1) % sample_every == 0:
        self.save_samples(f"samples_epoch_{epoch+1}.png", 4)
    
    return losses
  
  def save_samples(self, filename, num_samples=4):
    """Generate and save image samples"""
    samples = self.sample((num_samples, self.model.init_conv.in_channels, 32, 32))
    samples = samples.cpu().numpy()
    
    # Plot samples
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
    for i, sample in enumerate(samples):
      # Transpose for plotting (C,H,W) -> (H,W,C)
      sample = sample.transpose(1, 2, 0)
      if sample.shape[-1] == 1:  # Grayscale
        sample = sample.squeeze()
        axes[i].imshow(sample, cmap='gray')
      else:
        axes[i].imshow(sample)
      axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_with_diffusion(dataset, input_channels=3, batch_size=64, epochs=10, device=None):
  """
  Evaluate a dataset by training a diffusion model and measuring generation quality.
  
  Args:
    dataset: PyTorch dataset to evaluate
    input_channels: Number of input channels (3 for RGB, 1 for grayscale)
    batch_size: Batch size for training
    epochs: Number of training epochs
    device: Device to use (defaults to CUDA if available)
    
  Returns:
    Dictionary with evaluation metrics
  """
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # Create data loader
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  
  # Create model
  unet = UNet(in_channels=input_channels)
  diffusion = DiffusionModel(unet, device=device)
  
  # Train and evaluate
  print(f"Starting evaluation of dataset with {len(dataset)} samples")
  print(f"Training diffusion model for {epochs} epochs")
  
  losses = diffusion.evaluate_dataset(dataloader, num_epochs=epochs)
  
  # Generate final samples
  print("Generating final samples...")
  diffusion.save_samples("diffusion_final_samples.png", num_samples=8)
  
  # Return metrics
  return {
    "final_loss": losses[-1],
    "loss_curve": losses,
    "model": diffusion
  }


if __name__ == "__main__":
  # Example usage with CIFAR10
  
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])
  
  dataset = ds.Gi4eDataset(
    data_path="./datasets/GI4E",
    transform=transform,
    is_classification=True,
  )
  
  # Reduce dataset size for quick testing
  dataset = Subset(dataset, range(1000))
  
  results = evaluate_with_diffusion(
    dataset=dataset,
    input_channels=3,
    batch_size=4,
    epochs=5
  )
  
  print(f"Final loss: {results['final_loss']:.4f}")