import torch
import torch.nn as nn


class CfLinear(nn.Module):
    def __init__(self, input_dim=None, out_dim=None):
        super(CfLinear, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(self.input_dim, self.out_dim, bias=False)

    def forward(self, input):
        input_ = input.view([-1, self.input_dim])
        out = self.linear(input_)
        return out


class CfOrtho(nn.Module):
    def __init__(self, input_dim=None, out_dim=None):
        super(CfOrtho, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        init = 0.001 * torch.randn((self.out_dim, self.input_dim), device="cuda") + torch.eye(self.out_dim,
                                                                                              self.input_dim,
                                                                                              device="cuda")
        q, r = torch.qr(init)
        unflip = torch.diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        self.ortho_mat = nn.Parameter(q)

    def forward(self, input):
        with torch.no_grad():
            q, r = torch.qr(self.ortho_mat.data)
            unflip = torch.diag(r).sign().add(0.5).sign()
            q *= unflip[..., None, :]
            self.ortho_mat.data = q
        out = input @ self.ortho_mat.T
        return out
