import torch.nn as nn
import torch
from einops import einsum



class U_Mamba(nn.Module):
    def __init__(self, channels, width, height, hidden_size, rank, state_size, kernel_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.channels = channels
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.downscale_layers = nn.ModuleList([U_MambaBlockDownscale(channels=self.channels,
                                                 rank=self.rank,
                                                 state_size=self.state_size,
                                                 kernel_size=self.kernel_size,
                                                 hidden_size=self.hidden_size // 2 ** i,
                                                 width=width // 2 ** i,
                                                 height=height // 2 ** i) for i in range(num_layers)])
        
        self.upscale_layers = nn.ModuleList([U_MambaBlockUpscale(channels=self.channels,
                                            rank=self.rank,
                                            state_size=self.state_size,
                                            kernel_size=self.kernel_size,
                                            hidden_size=self.hidden_size // (2 ** i),
                                            width=width // 2 ** i,
                                            height=height // 2 ** i) for i in range(num_layers, 0, -1)])
        

            
    def forward(self, x):
        activation_history = []

        # Pass through the layers, save for residual 

        for layer in self.downscale_layers:

            x = layer(x)
            activation_history.append(x)

        for layer in self.upscale_layers:

            x = x + activation_history.pop()
            x = layer(x)
        
        return x


class U_MambaBlockUpscale(nn.Module):
    def __init__(self, channels, width, height, hidden_size, rank, state_size, kernel_size):
        super().__init__()
        self.layer = U_MambaLayer(channels=channels,
                                hidden_size=hidden_size,
                                width=width,
                                height=height, 
                                rank=rank,
                                state_size=state_size,
                                kernel_size=kernel_size)
        self.upscale_conv = nn.ConvTranspose3d(in_channels=channels,
                                        out_channels=channels,
                                        kernel_size=2,
                                        stride = 2,
                                        padding=0)
        

    def forward(self, x):
        x = self.layer(x)
        x = self.upscale_conv(x)
        return x

class U_MambaBlockDownscale(nn.Module):
    def __init__(self, channels, width, height, hidden_size, rank, state_size, kernel_size):
        super().__init__()
        self.layer = U_MambaLayer(channels=channels,
                                hidden_size=hidden_size,
                                width=width,
                                height=height,
                                rank=rank,
                                state_size=state_size,
                                kernel_size=kernel_size)
        
        self.downscale_conv = nn.Conv3d(in_channels=channels,
                                        out_channels=channels,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1)//2,
                                        stride=2)
        

    def forward(self, x):
        x = self.layer(x)
        x = self.downscale_conv(x)
        return x

class U_MambaLayer(nn.Module):
    def __init__(self, channels, hidden_size, width, height, rank, state_size, kernel_size):
        super().__init__()
        self.channels = channels
        self.width = width
        self.height = height
        self.hidden_size = hidden_size
        self.rank = rank
        self.state_size = state_size
        self.kernel_size = kernel_size
        signal_length = width * height * hidden_size

        # this convolution is a "same" 3d convolution 
        self.conv = nn.Conv3d(in_channels=self.channels,
                                        out_channels=self.channels,
                                        kernel_size=self.kernel_size,
                                        padding=(self.kernel_size - 1)//2) 
        
        self.conv1d = nn.Conv1d(in_channels=self.channels,
                              out_channels=self.channels,
                              kernel_size=self.kernel_size,
                              padding=(self.kernel_size - 1)//2)
        
        self.linear_dt = nn.Linear(rank, signal_length)
        self.leaky_relu = nn.LeakyReLU()

        self.layer_norm = nn.LayerNorm(signal_length)

        self.linear_left = nn.Linear(signal_length, signal_length)
        self.linear_right = nn.Linear(signal_length, signal_length)


        self.linear_x = nn.Linear(signal_length, rank + 2 * state_size) 
        self.linear_out = nn.Linear(signal_length, signal_length)

        self.A = torch.arange(1, state_size + 1).unsqueeze(0).repeat(signal_length, 1)
        self.A_log = torch.log(self.A + 1e-15)
        self.D = nn.Parameter(torch.ones(signal_length))

    def forward(self, x):
        batch_size, channels, width, height, hidden_size = x.shape
                
        # shape (batch_size, channels, width, height, dim)
        x_conv = self.conv(x)
        x_conv = x_conv + self.leaky_relu(x_conv)

        x = x_conv + x
        
        #again, times 2 from the paper
        x = self.conv(x)
        x = x + self.leaky_relu(x)
        x = x + x_conv

        #shape here should be the same, [batch_size, channels, width, height, dim]
        # flatten to batch, channels, length 
        x = x.view(batch_size, channels, -1)

        # layer norm
        x = self.layer_norm(x)

        
        # left branch, linear -> 1d conv -> SSM
        x_left = self.linear_left(x)
        x_left = self.conv1d(x_left)
        x_left = torch.nn.functional.silu(x_left)
        x_left = self.ssm(x_left)

        # right branch, linear -> silu
        x_right = self.linear_right(x)
        x_right = torch.nn.functional.silu(x_right) 

        # elementwise product
        x = x_left * x_right

        # linear out   
        x = self.linear_out(x)

        #reshape back to batch, channels, width, height, dim


        x = x.view(batch_size, channels, width, height, hidden_size)

        return x


    def ssm(self, x):
        hidden_size, n = self.A_log.shape

        # compute A, D 
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # get delta B and C from input (specifically for mamba, not for general selective scan)
        x_dbl = self.linear_x(x)
        delta, B, C = x_dbl.split(split_size=[self.rank, n, n], dim=-1)
        delta = torch.nn.functional.softplus(self.linear_dt(delta))

        # selective scan 
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        batch_size, sequence_length, hidden_size = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'batch seq_length hidden_size, hidden_size n -> batch seq_length hidden_size n'))
        deltaB_u = einsum(delta, B, u, 'batch seq_length hidden_size, batch seq_length n, batch seq_length hidden_size -> batch seq_length hidden_size n')

        # the state 
        x = torch.zeros((batch_size, hidden_size, n), device=deltaA.device)

        ys = []

        # selective scan (x = A * x + B * u, y = C * x)
        for i in range(sequence_length):
            x = deltaA[:, i, :] * x + deltaB_u[:, i, :]
            y = (x * C[:, i, :].unsqueeze(1)).sum(-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)

        # u * D to y
        y = y + u * D.unsqueeze(0).unsqueeze(0)
        return y
    




# channels = 3
# hidden_size = 16
# width = 16
# height = 16
# rank = 4
# state_size = 4
# kernel_size = 3

# model = U_Mamba(channels=channels, width=width, height=height, hidden_size=hidden_size, rank=rank, state_size=state_size, kernel_size=kernel_size, num_layers=3)
# model(x).shape == x.shape