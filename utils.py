import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm
import functools
from torch.nn import Parameter as P

def snconv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return SpectralNorm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, bias=bias))

def snlinear(in_features, out_features):
    return SpectralNorm(nn.Linear(in_features=in_features, out_features=out_features))

class Attention(nn.Module):
  def __init__(self, in_dim):
      super(Attention, self).__init__()
      self.chanel_in = in_dim

      self.query_conv = snconv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, padding=0, bias=False)
      self.key_conv = snconv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, padding=0, bias=False)
      self.value_conv = snconv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, padding=0, bias=False)

      self.gamma = P(torch.zeros(1))

      self.softmax = nn.Softmax(dim=-1)
  def forward(self, x):
      m_batchsize, C, height, width = x.size()
      proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
      proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
      energy = torch.bmm(proj_query, proj_key)
      attention = self.softmax(energy)
      proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

      out = torch.bmm(proj_value, attention.permute(0, 2, 1))
      out = out.view(m_batchsize, C, height, width)

      out = self.gamma*out + x
      #out = F.avg_pool2d(out, out.size()[2:4])

      return out

class GBlock(nn.Module):
  def __init__(self, in_channels, out_channels, upsample=None, channel_ratio=4):
    super(GBlock, self).__init__()
    
    self.in_channels, self.out_channels = in_channels, out_channels
    self.hidden_channels = self.in_channels // channel_ratio
    
    # Conv layers
    self.conv1 = snconv3d(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv3 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv4 = snconv3d(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
    # Batchnorm layers
    self.bn1 = nn.BatchNorm2d(self.in_channels)
    self.bn2 = nn.BatchNorm2d(self.hidden_channels)
    self.bn3 = nn.BatchNorm2d(self.hidden_channels)
    self.bn4 = nn.BatchNorm2d(self.hidden_channels)
    # upsample layers
    self.upsample = upsample
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x):
    # Project down to channel ratio
    h = self.conv1(self.activation(self.bn1(x)))
    # Apply next BN-ReLU
    h = self.activation(self.bn2(h))
    if self.in_channels != self.out_channels:
      x = x[:, :self.out_channels]   
    # Upsample both h and x at this point
    if self.upsample:
      h = self.upsample(h)
      x = self.upsample(x)
    # 3x3 convs
    h = self.conv2(h)
    h = self.conv3(self.activation(self.bn3(h)))
    # Final 1x1 conv
    h = self.conv4(self.activation(self.bn4(h)))
    return h + x

class DBlock(nn.Module):
  def __init__(self, in_channels, out_channels, wide=True, preactivation=True,
               downsample=None, channel_ratio=4):
    super(DBlock, self).__init__()
    self.in_channels, self.out_channels = in_channels, out_channels
    # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
    self.hidden_channels = self.out_channels // channel_ratio

    self.preactivation = preactivation
    self.activation = nn.ReLU(inplace=True)
    self.downsample = downsample
        
    # Conv layers
    self.conv1 = snconv3d(self.in_channels, self.hidden_channels, 
                                 kernel_size=1, padding=0)
    self.conv2 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv3 = snconv3d(self.hidden_channels, self.hidden_channels)
    self.conv4 = snconv3d(self.hidden_channels, self.out_channels, 
                                 kernel_size=1, padding=0)
                                 
    self.learnable_sc = True if (in_channels != out_channels) else False
    if self.learnable_sc:
      self.conv_sc = snconv3d(in_channels, out_channels - in_channels, 
                                     kernel_size=1, padding=0)
  def shortcut(self, x):
    if self.downsample:
      x = self.downsample(x)
    if self.learnable_sc:
      x = torch.cat([x, self.conv_sc(x)], 1)    
    return x
    
  def forward(self, x):
    # 1x1 bottleneck conv
    h = self.conv1(F.relu(x))
    # 3x3 convs
    h = self.conv2(self.activation(h))
    h = self.conv3(self.activation(h))
    # relu before downsample
    h = self.activation(h)
    # downsample
    if self.downsample:
      h = self.downsample(h)     
    # final 1x1 conv
    h = self.conv4(h)
    return h + self.shortcut(x)