from torch import nn
from torch.nn import functional as F
from utils import *
import torch


class LatentDeformator(nn.Module):
    def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024,
                 type='ortho', random_init=False, bias=False):
        super(LatentDeformator, self).__init__()
        self.type = type
        self.shift_dim = shift_dim
        self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
        self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

        if self.type in ['linear', 'proj']:
            self.linear = nn.Linear(self.input_dim, self.out_dim, bias=bias)
            if random_init:
                self.linear.weight.data = 0.1 * torch.randn_like(self.linear.weight.data)

        elif self.type == 'ortho':
            init = 0.001 * torch.randn(
                (self.out_dim, self.input_dim), device="cuda"
            ) + torch.eye(self.out_dim, self.input_dim, device="cuda")

            q, r = torch.qr(init)
            unflip = torch.diag(r).sign().add(0.5).sign()
            q *= unflip[..., None, :]
            self.ortho_mat = nn.Parameter(q)

        elif self.type == 'random':
            self.linear = torch.empty([self.out_dim, self.input_dim])
            nn.init.orthogonal_(self.linear)

    def forward(self, input):
        if self.type == 'linear':
            out = self.linear(input)

        elif self.type == 'ortho':
            with torch.no_grad():
                q, r = torch.qr(self.ortho_mat.data)
                unflip = torch.diag(r).sign().add(0.5).sign()
                q *= unflip[..., None, :]
                self.ortho_mat.data = q
            out = input @ self.ortho_mat.T

        elif self.type == 'random':
            self.linear = self.linear.to(input.device)
            out = F.linear(input, self.linear)

        flat_shift_dim = np.product(self.shift_dim)
        if out.shape[1] < flat_shift_dim:
            padding = torch.zeros([out.shape[0], flat_shift_dim - out.shape[1]]).cuda()
            out = torch.cat([out, padding], dim=1)
        elif out.shape[1] > flat_shift_dim:
            out = out[:, :flat_shift_dim]

        # handle spatial shifts
        try:
            out = out.view([-1] + self.shift_dim)
        except Exception:
            pass

        return out


def normal_projection_stat(x):
    x = x.view([x.shape[0], -1])
    direction = torch.randn(x.shape[1], requires_grad=False).cuda()
    direction = direction / torch.norm(direction)
    projection = torch.matmul(x, direction)

    std, mean = torch.std_mean(projection)
    return std, mean
