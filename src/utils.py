import os
import sys
import json
import torch
import logging
from scipy.stats import truncnorm


def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)
    else:
        return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)


def is_conditional(G):
	return 'biggan' in G.__class__.__name__.lower()


def one_hot(dims, value, indx):
	vec = torch.zeros(dims)
	vec[indx] = value
	return vec


def save_command_run_params(args):
	os.makedirs(args.out, exist_ok=True)
	with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
		json.dump(args.__dict__, args_file)
	with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
		command_file.write(' '.join(sys.argv))
		command_file.write('\n')


def truncated_noise(size, truncation=1.0):
	return truncnorm.rvs(-truncation, truncation, size=size)

import torch
import numpy as np


def torch_expm(A):
    n_A = A.shape[0]
    A_fro = torch.sqrt(A.abs().pow(2).sum(dim=(1, 2), keepdim=True))

    # Scaling step
    maxnorm = torch.tensor([5.371920351148152], dtype=A.dtype, device=A.device)
    zero = torch.tensor([0.0], dtype=A.dtype, device=A.device)
    n_squarings = torch.max(zero, torch.ceil(torch_log2(A_fro / maxnorm)))
    A_scaled = A / 2.0 ** n_squarings
    n_squarings = n_squarings.flatten().type(torch.int64)

    # Pade 13 approximation
    U, V = torch_pade13(A_scaled)
    P = U + V
    Q = -U + V
    R, _ = torch.solve(P, Q)

    # Unsquaring step
    res = [R]
    for i in range(int(n_squarings.max())):
        res.append(res[-1].matmul(res[-1]))
    R = torch.stack(res)
    expmA = R[n_squarings, torch.arange(n_A)]
    return expmA[0]


def torch_log2(x):
    return torch.log(x) / np.log(2.0)


def torch_pade13(A):
    b = torch.tensor([64764752532480000., 32382376266240000., 7771770303897600.,
                      1187353796428800., 129060195264000., 10559470521600.,
                      670442572800., 33522128640., 1323241920., 40840800.,
                      960960., 16380., 182., 1.], dtype=A.dtype, device=A.device)

    ident = torch.eye(A.shape[1], dtype=A.dtype).to(A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(A,
                     torch.matmul(A6, b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 +
                     b[3] * A2 + b[1] * ident)
    V = torch.matmul(A6, b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 +\
        b[0] * ident
    return U, V


def make_ortho(a, dim):
    mat_log = torch.zeros([dim, dim])
    it = 0
    for i in range(dim):
        for j in range(i + 1, dim, 1):
            mat_log[i, j] = a[it]
            mat_log[j, i] = -a[it]
            it += 1
    return torch_expm(mat_log.unsqueeze(0))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logging.info(net)
    logging.info('Total number of parameters: %d' % num_params)