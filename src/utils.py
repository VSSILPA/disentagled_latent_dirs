import os
import sys
import json
import torch
import fnmatch
import shutil
from typing import List, Tuple
from scipy.stats import truncnorm
from torch.utils.data import Dataset
TINY = 1e-12

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

class NoiseDataset(Dataset):
    def __init__(self, num_samples, z_dim):
        self.num_samples = num_samples
        self.z_dim = z_dim
        self.data = torch.randn(num_samples, z_dim)

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)
    else:
        return torch.from_numpy(truncated_noise([batch] + dim, truncation)).to(torch.float)


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

    def mse(predicted, target):
        """mean square error """
        predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted  # (n,)->(n,1)
        target = target[:, None] if len(target.shape) == 1 else target  # (n,)->(n,1)
        err = predicted - target
        err = err.T.dot(err) / len(err)
        return err[0, 0]  # value not array

def mse(predicted, target):
    """mean square error """
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted  # (n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target  # (n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0]  # value not array

def rmse(predicted, target):
    """ root mean square error """
    return np.sqrt(mse(predicted, target))

def nmse(predicted, target):
    """ normalized mean square error """
    return mse(predicted, target) / np.var(target)

def nrmse(predicted, target):
    """ normalized root mean square error """
    return rmse(predicted, target) / np.std(target)

def normalize(X, mean=None, stddev=None, useful_features=None, remove_constant=True):
    calc_mean, calc_stddev = False, False

    if mean is None:
        mean = np.mean(X, 0)  # training set
        calc_mean = True

    if stddev is None:
        stddev = np.std(X, 0)  # training set
        calc_stddev = True
        useful_features = np.nonzero(stddev)[0]  # inconstant features, ([0]=shape correction)

    if remove_constant and useful_features is not None:
        X = X[:, useful_features]
        if calc_mean:
            mean = mean[useful_features]
        if calc_stddev:
            stddev = stddev[useful_features]

    X_zm = X - mean
    X_zm_unit = X_zm / stddev

    return X_zm_unit, mean, stddev, useful_features

def norm_entropy(p):
    """p: probabilities """
    n = p.shape[0]
    return - p.dot(np.log(p + TINY) / np.log(n + TINY))

def entropic_scores(r):
    """r: relative importances"""
    r = np.abs(r)
    ps = r / np.sum(r, axis=0)  # 'probabilities'
    hs = [1 - norm_entropy(p) for p in ps.T]
    return hs

def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


def list_dir_recursively_with_ignore(dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False) -> \
        List[Tuple[str, str]]:
    """List all files recursively in a given directory while ignoring given file and directory names.
    Returns list of tuples containing both absolute and relative paths."""
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result

