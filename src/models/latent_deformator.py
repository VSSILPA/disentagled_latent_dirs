from torch import nn
from torch.nn import functional as F
from utils import *
import torch


class LatentDeformator(nn.Module):
    def __init__(self, shift_dim, input_dim=None, out_dim=None, inner_dim=1024,
                 type='ortho', random_init=False, bias=True):
        super(LatentDeformator, self).__init__()
        self.type = type
        self.shift_dim = shift_dim
        self.input_dim = input_dim if input_dim is not None else np.product(shift_dim)
        self.out_dim = out_dim if out_dim is not None else np.product(shift_dim)

        if self.type == 'fc':
            self.fc1 = nn.Linear(self.input_dim, inner_dim)
            self.bn1 = nn.BatchNorm1d(inner_dim)
            self.act1 = nn.ELU()

            self.fc2 = nn.Linear(inner_dim , inner_dim)
            self.bn2 = nn.BatchNorm1d(inner_dim)
            self.act2 = nn.ELU()

            self.fc3 = nn.Linear(inner_dim, inner_dim)
            self.bn3 = nn.BatchNorm1d(inner_dim)
            self.act3 = nn.ELU()

            self.fc4 = nn.Linear(inner_dim, self.out_dim)

        elif self.type in ['linear', 'proj']:
            self.linear = nn.Linear(self.input_dim, self.out_dim,bias=bias)
            self.linear.weight.data = torch.zeros_like(self.linear.weight.data)

            min_dim = int(min(self.input_dim, self.out_dim))
            self.linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            if random_init:
                self.linear.weight.data = 0.1 * torch.randn_like(self.linear.weight.data)

        elif self.type == 'ortho-synthethic':
            init = 0.001 * torch.randn(
                (self.out_dim, self.input_dim), device="cuda"
            ) + torch.eye(self.out_dim, self.input_dim, device="cuda")

            q, r = torch.qr(init)
            unflip = torch.diag(r).sign().add(0.5).sign()
            q *= unflip[..., None, :]
            self.ortho_mat = nn.Parameter(q)
        elif self.type == 'ortho-natural':
            self.log_mat_half = nn.Parameter((1.0 if random_init else 0.001) * torch.randn(
                [self.input_dim, self.input_dim], device='cuda'), True)

        elif self.type == 'random':
            self.linear = torch.empty([self.out_dim, self.input_dim])
            nn.init.orthogonal_(self.linear)

    def forward(self, input):
        original_shape = input.shape
        if self.type == 'id':
            return input

        input = input.view([-1, self.input_dim])
        if self.type == 'fc':
            x1 = self.fc1(input)
            x = self.act1(self.bn1(x1))

            x2 = self.fc2(x)
            x = self.act2(self.bn2(x2 + x1))

            x3 = self.fc3(x)
            x = self.act3(self.bn3(x3 + x2 + x1))

            out = self.fc4(x) + input
        elif self.type == 'linear':
            out = self.linear(input)
        elif self.type == 'proj':
            input_norm = torch.norm(input, dim=1, keepdim=True)
            out = self.linear(input)
            out = (input_norm / torch.norm(out, dim=1, keepdim=True)) * out
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
