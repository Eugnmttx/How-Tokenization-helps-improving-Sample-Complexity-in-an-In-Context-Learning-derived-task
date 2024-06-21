import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_mixer(nn.Module):
    def __init__(
        self, dim_token, dim_feature, hidden_dim_token, hidden_dim_feature, out_dim, norm='std'
    ):
        """
        MultiLayer Perceptron Mixer

        Args:
            input_dim: The input dimension.
            nn_dim: The number of hidden neurons per layer.
            out_dim: The output dimension.
            num_layers: The number of layers.
            bias: True for adding bias.
            norm: Scaling factor for the readout layer.
        """
        super().__init__()

        self.dim_token = dim_token
        self.dim_feature = dim_feature
        self.out_dim = out_dim
        self.hidden_dim_token = hidden_dim_token
        self.hidden_dim_feature = hidden_dim_feature

        self.norm_token = self.hidden_dim_token**0.5
        self.norm_feature = self.hidden_dim_feature**0.5
        
        if norm == 'std':
            self.norm = self.hidden_dim_feature*self.hidden_dim_token**0.5
        elif norm == 'mf':
            self.norm = self.hidden_dim_feature*self.hidden_dim_token
        else:
            raise ValueError('Model type is not well specified, it should be "mf" or "std"') from None

        self.W1=nn.Parameter(
            torch.randn( self.hidden_dim_token, self.dim_token)
        )
        self.b1=nn.Parameter(
            torch.randn(self.hidden_dim_token)
        )
        self.W2=nn.Parameter(
            torch.randn( self.hidden_dim_token, self.hidden_dim_token)
        )
        self.b2=nn.Parameter(
            torch.randn( self.hidden_dim_token)
        )
        self.W3=nn.Parameter(
            torch.randn( self.hidden_dim_feature, self.dim_feature)
        )
        self.b3=nn.Parameter(
            torch.randn( self.hidden_dim_feature)
        )
        self.W4=nn.Parameter(
            torch.randn( self.out_dim, self.hidden_dim_feature * self.hidden_dim_token)
        )

    def forward(self, x):
        """
        Args:
            x: input, tensor of size (batch_size, *, input_dim).
        
        Returns:
            Output of a multilayer perceptron, tensor of size (batch_size, *, out_dim)
        """
        x = F.linear( x, self.W1, self.b1)
        x = nn.ReLU()(x)
        x = F.linear( x, self.W2, self.b2) / self.norm_token
        x = F.linear( x.transpose(1,2), self.W3, self.b3).transpose(1,2) / self.norm_feature
        x = nn.ReLU()(x)
        x = x.flatten(1,2)
        x = F.linear( x, self.W4) / self.norm
        return x
