import random
import math

import torch
import numpy as np
import scipy.linalg

def add_full(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    added to every element
    """
    return lambda x: x + (torch.ones_like(x) * r)


def multiply(r):
    return lambda x: x * r


def tensor_multiply(op):
    return lambda x: torch.tensordot(op, x, dims=1)


def add_sparse(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    added to 25% of the elements
    """
    return lambda x: x + ((torch.rand_like(x) < 0.05) * r)


def add_noise(r):
    """
    Return a fn that adds Gaussian noise mutliplied by r to x
    """

    return lambda x: x + (torch.randn_like(x) * r)


def subtract_full(r):
    """
    Return a fn that takes in a latent tensor and returns a tensor of the same shape, but with the value r
    subtracted from every element
    """
    return lambda x: x - (torch.ones_like(x) * r)


def threshold(r):
    def thresh(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: y if abs(y) >= r else 0)
        if device != -1:
            x = x.to(device)
        return x
    return thresh


def soft_threshold(r):
    """
    Return a fn that applies soft thresholding to x
    In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    source: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    """
    def fn(x):
        x = x / x.abs() * torch.maximum(x.abs() - r, torch.zeros_like(x))
        return x
    return fn


def soft_threshold2(r):
    """
    Return a fn that applies soft thresholding to x
    In soft thresholding, values less than r are set to 0 and values greater than r are shrunk towards zero
    source: https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html
    """
    def fn(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: 0 if abs(y) < r else y*(1-r))
        x = x.to(device)
        return x
    return fn


def inversion(r):
    def foo(x):
        device = x.get_device()
        x = x.cpu()
        x = x.apply_(lambda y: 1./r - y)
        x = x.to(device)
        return x
    return foo


def inversion2():
    return lambda x: -1 * x


def log(r):
    return lambda x: torch.log(x)


def power(r):
    return lambda x: torch.pow(x, r)

def add_dim(r, dim, i):
    def foo(x):
        """
        Add value r to the given dim at index i
        dim = 0 means adding in "z" dimension (shape = 4)
        dim = 1 means adding to row i
        dim = 2 means adding to column i
        @return: modified x
        """
        if dim == 0:
            for a in range(x.shape[0]):
                x[a, i, i] += r
        elif dim == 1:
            x[:, i:i + 1] += r
        elif dim == 2:
            x[:, :, i:i + 1] += r
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo

def add_rand_cols(r, k):
    """
    Return a fn that will add value 'r' to a fraction of the cols of a tensor
    Assume x is a 3-tensor and rows refer to the third dimension
    k should be between 0 and 1
    """
    def foo(x):
        dim = x.shape[1]
        cols = random.sample(range(dim), int(k * dim))
        for col in cols:
            x[:, :, col:col + 1] += r
        return x

    return foo


def add_rand_rows(r, k):
    """
    Return a fn that will add value 'r' to a fraction of the rows of a tensor
    Assume x is a 3-tensor and rows refer to the second dimension
    k should be between 0 and 1
    """
    def foo(x):
        dim = x.shape[2]
        rows = random.sample(range(dim), int(k * dim))
        for row in rows:
            x[:, row:row + 1] += r
        return x

    return foo


def invert_dim(r, dim, i):
    def foo(x):
        """
        Apply inversion (1/r. - x) at the given dim at index i
        dim = 0 means applying in "z" dimension (shape = 4)
        dim = 1 means applying to row i
        dim = 2 means applying to column i
        @return: modified x
        """
        invert = inversion(r)
        if dim == 0:
            for a in range(x.shape[0]):
                x[a, i, i] = invert(x[a, i, i])
        elif dim == 1:
            x[:, i:i + 1] = invert(x[:, i:i + 1])
        elif dim == 2:
            x[:, :, i:i + 1] += invert(x[:, :, i:i + 1])
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo


def apply_to_dim(func, r, dim, i):
    def foo(x):
        """
        Apply func at the given dim at i
        dim = 0 means applying in "z" dimension (shape = 4)
        dim = 1 means applying to row i
        dim = 2 means applying to column i
        @return: modified x
        """
        fn = func(r)
        if dim == 0:
            for a in range(x.shape[0]):
                try:
                    x[a, i[0], i[1]] = fn(x[a, i[0], i[1]])
                except TypeError:
                    x[a, i, i] = fn(x[a, i, i])
        elif dim == 1:
            try:
                x[:, i[0]:i[1]] = fn(x[:, i[0]:i[1]])
            except TypeError:
                x[:, i:i + 1] = fn(x[:, i:i + 1])
        elif dim == 2:
            try:
                x[:, :, i[0]:i[1]] = fn(x[:, :, i[0]:i[1]])
            except TypeError:
                x[:, :, i:i + 1] = fn(x[:, :, i:i + 1])
        else:
            raise NotImplementedError(f"Cannot apply to dimension {dim}")
        return x

    return foo


def apply_sparse(func, sparsity):
    """
    return a function that applies the given function a random fraction of the elements, as determined by 'sparsity'
    0 < sparsity < 1
    """
    def fn(x):
        mask = torch.rand_like(x) < sparsity
        x = (x * ~mask) + (func(x) * mask)
        return x
    return fn


def add_normal(r):
    """
    Add a 2D normal gaussian (bell curve) to the center of the tensor
    """
    def foo(x):
        # chatgpt wrote this
        # Define the size of the matrix
        size = 64

        # Generate grid coordinates centered at (0,0)
        a = np.linspace(-5, 5, size)
        b = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(a, b)

        # Standard deviation
        sigma = 1.5  # You can adjust this value as desired

        # Generate a 2D Gaussian distribution with peak at the center and specified standard deviation
        Z = np.exp(-0.5 * ((X / sigma) ** 2 + (Y / sigma) ** 2)) / (2 * np.pi * sigma ** 2)
        Z *= r
        Z = torch.from_numpy(Z).to(x.get_device())

        for i in range(x.shape[0]):
            x[i] += Z
        return x
    return foo


def tensor_exp(r):
    """
    Return a fn that computes the matrix exponential of a given tensor
    """
    def foo(x):
        device = x.get_device()
        x = x.cpu().numpy()
        x = scipy.linalg.expm(x)
        x = torch.from_numpy(x).to(device)
        return x
    return foo


def rotate_z(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "z" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c, -1 * s, 0, 0],
            [s,    c,   0, 0],
            [0,    0,   1, 0],
            [0,    0,   0, 1]
        ]
        op = torch.tensor(rotation_matrix).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def rotate_x(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "x" axis
    """
    def fn(x):
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [1, 0,   0,    0],
            [0, c, -1 * s, 0],
            [0, s,   c,    0],
            [0, 0,   0,    1]
        ]
        op = torch.tensor(rotation_matrix).to(x)  # this sets the device and dtype !
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def rotate_y(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "y" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c,      0, s, 0],
            [0,      1, 0, 0],
            [-1 * s, 0, c, 0],
            [0,      0, 0, 1]
        ]
        op = torch.tensor(rotation_matrix).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def rotate_y2(r):
    """
    Return a fn that rotates a 3-tensor by r radians
    Rotates along "y" axis
    """
    def fn(x):
        device = x.get_device()
        c = math.cos(r)
        s = math.sin(r)
        rotation_matrix = [
            [c,      0, 0, s],
            [0,      1, 0, 0],
            [0,      0, 1, 0],
            [-1 * s, 0, 0, c]
        ]
        op = torch.tensor(rotation_matrix).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def make_rotation_matrix(angle, plane):
    """
    ChatGPT wrote this
    Generate a 4x4 rotation matrix for a specific plane in 4D space using PyTorch.

    Args:
        angle (float): Rotation angle in radians (as a torch scalar or float).
        plane (tuple): A pair of axes defining the plane (e.g., (0, 1) for x_1x_2).

    Returns:
        torch.Tensor: The 4x4 rotation matrix as a PyTorch tensor.
    """
    dim = 4
    matrix = torch.eye(dim)  # Initialize as identity matrix
    i, j = plane
    angle = torch.tensor(angle)

    # Rotation in the specified plane
    matrix[i, i] = torch.cos(angle)
    matrix[j, j] = torch.cos(angle)
    matrix[i, j] = -torch.sin(angle)
    matrix[j, i] = torch.sin(angle)

    return matrix


def make_six_plane_rotation_matrix(angles):
    """
    ChatGPT wrote this
    Generate the combined 4x4 rotation matrix for all six planes using PyTorch.

    Args:
        angles (list of float): List of six rotation angles in radians (as torch tensors or floats).

    Returns:
        torch.Tensor: The combined 4x4 rotation matrix as a PyTorch tensor.
    """
    planes = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    combined_matrix = torch.eye(4)  # Start with identity matrix
    for angle, plane in zip(angles, planes):
        # Multiply the matrices (from left to right)
        combined_matrix = torch.matmul(combined_matrix, make_rotation_matrix(angle, plane))
    return combined_matrix


def six_plane_rotation(angles):
    def rotate(x):
        matrix = make_six_plane_rotation_matrix(angles)
        matrix = matrix.to(x)
        return torch.tensordot(matrix, x, dims=1)
    return rotate


def reflect(r):
    """
    Return a fn that reflects across the given dimension r
    r can be 0, 1, 2, or 3
    """
    def fn(x):
        device = x.get_device()
        op = torch.eye(4)  # identity matrix
        op[r, r] *= -1
        op = op.to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def hadamard1(r):
    def fn(x):
        device = x.get_device()
        h = scipy.linalg.hadamard(4)
        op = torch.tensor(h).to(torch.float32).to(device)
        x = torch.tensordot(op, x, dims=1)
        return x
    return fn


def hadamard2(r):
    def fn(x):
        device = x.get_device()
        h = scipy.linalg.hadamard(64)
        op = torch.tensor(h).to(torch.float32).to(device)
        x = torch.tensordot(x, op, dims=[[1], [1]])
        return x
    return fn


def apply_both(fn1, fn2, r):
    def fn(x):
        return fn1(fn2(r))
    return fn



def normalize(func):
    """
    First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)  # first apply the network bending function
        # then normalize the result
        max = x.abs().max()
        x = x / max
        return x
    return fn


def normalize2(func):
    """
    First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)  # first apply the network bending function
        # then normalize the result
        x = torch.nn.functional.normalize(x, dim=0)
        return x
    return fn


def normalize3(func):
    """
        First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)
        x = x - x.mean()
        return x
    return fn


def normalize4(func, dim=0):
    """
        First apply func to the latent tensor, then normalize the result
    """
    def fn(x):
        x = func(x)
        x = x - torch.mean(x, dim=dim, keepdim=True)
        return x
    return fn
