import torch.nn as nn
import torch
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-15):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * self.weight
        return output


class Mamba(nn.Module):
    def __init__(self, model_size, vocab_size, rank, state_size, kernel_size, num_layers):
        super().__init__()
        self.hidden_size = model_size * 2
        self.model_size = model_size
        self.rank = rank
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.model_size)
        self.linear = nn.Linear(model_size, vocab_size)
        self.norm = RMSNorm(eps=1e-15, hidden_size=self.model_size)
        self.layers = nn.ModuleList([MambaBlock(model_size=self.model_size,
                                                 hidden_size=self.hidden_size,
                                                 rank=self.rank,
                                                 state_size=self.state_size,
                                                 kernel_size=self.kernel_size) for i in range(num_layers)])
    def forward(self, input_ids):
        # Get embeddings for input_ids
        x = self.embedding(input_ids)

        # Pass through the layers
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        
        return self.linear(x)


class MambaBlock(nn.Module):
    def __init__(self, model_size, hidden_size, rank, state_size, kernel_size):
        super().__init__()
        self.layer = MambaLayer(model_size=model_size,
                                hidden_size=hidden_size,
                                rank=rank,
                                state_size=state_size,
                                kernel_size=kernel_size)
        self.norm = RMSNorm(model_size)

    def forward(self, x):
        return x + self.norm(self.layer(x)) 

class MambaLayer(nn.Module):
    def __init__(self, model_size, hidden_size, rank, state_size, kernel_size):
        super().__init__()
        self.model_size = model_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=self.hidden_size,
                              out_channels=self.hidden_size,
                              kernel_size=self.kernel_size,
                              groups=self.hidden_size,
                              padding=self.kernel_size - 1)
        
        self.linear_dt = nn.Linear(rank, hidden_size)
        
        self.linear_in = nn.Linear(model_size, hidden_size * 2)
        self.linear_x = nn.Linear(hidden_size, rank + 2 * state_size) 
        self.linear_out = nn.Linear(hidden_size, self.model_size)

        self.A = torch.arange(1, state_size + 1).unsqueeze(0).repeat(hidden_size, 1)
        self.A_log = torch.log(self.A + 1e-15)
        self.D = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.shape
                
        # shape (batch_size, seq_length, 2 * d_inner)
        x_with_residual = self.linear_in(x)

        # split x_with_residual into two tensors of shape (batch_size, seq_length, d_inner)
        x, residual = x_with_residual.split(split_size=[self.hidden_size, self.hidden_size], dim=-1)

        # rearrange x to shape (batch_size, d_inner, seq_length)
        x = x.permute(0, 2, 1)

        # convolution to x, then slice it to match the original sequence length
        x = self.conv(x)[:, :, :seq_length]

        # to shape (batch_size, seq_length, d_inner)
        x = x.permute(0, 2, 1)

        # apply the silu 
        x = torch.nn.functional.silu(x)

        # selective state model mechanism to 'x
        y = self.ssm(x)

        y = y * torch.nn.functional.silu(residual)

        output = self.linear_out(y)
        return output

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
