import numpy as np
import torch


def _is_torch(x):
    return isinstance(x, torch.Tensor)


def norm(q):
    # q: (batch_size, 4)
    if _is_torch(q):
        return torch.sqrt(torch.sum(q**2, dim=1, keepdim=True))
    return np.sqrt(np.sum(q**2, axis=1, keepdims=True))


def normalize(q):
    # q: (batch_size, 4)
    if _is_torch(q):
        return q / (norm(q) + 1e-20)
    return q / (norm(q) + 1e-20)


def conjugate(q):
    # q: (batch_size, 4)
    if _is_torch(q):
        return torch.cat([q[:, 0:1], -q[:, 1:4]], dim=1)
    return np.concatenate([q[:, 0:1], -q[:, 1:4]], axis=1)


def inverse(q):
    # q: (batch_size, 4)
    if _is_torch(q):
        return conjugate(q) / (torch.sum(q**2, dim=1, keepdim=True) + 1e-20)
    return conjugate(q) / (np.sum(q**2, axis=1, keepdims=True) + 1e-20)


def from_vectors(a, b):
    # get the quaternion from two 3D vectors, such that b = qa.
    # a: (batch_size, 3)
    # b: (batch_size, 3)
    # note: a and b don't need to be unit vectors.
    if _is_torch(a):
        q = torch.empty(a.shape[0], 4, device=a.device, dtype=a.dtype)
        q[:, 0] = torch.sqrt(torch.sum(a**2, dim=1)) * torch.sqrt(
            torch.sum(b**2, dim=1)
        ) + torch.sum(a * b, dim=1)
        q[:, 1:] = torch.cross(a, b)
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        q = np.empty((a.shape[0], 4), dtype=a.dtype)
        q[:, 0] = np.sqrt(np.sum(a**2, axis=1)) * np.sqrt(
            np.sum(b**2, axis=1)
        ) + np.sum(a * b, axis=1)
        q[:, 1:] = np.cross(a, b)

    q = normalize(q)
    return q


def from_axis_angle(axis, angle):
    # get the quaternion from axis-angle representation
    # axis: (batch_size, 3)
    # angle: (batch_size, 1), in radians
    if _is_torch(axis):
        axis_n = normalize(axis)
        q = torch.empty(axis.shape[0], 4, device=axis.device, dtype=axis.dtype)
        q[:, 0] = torch.cos(angle / 2)
        q[:, 1:] = axis_n * torch.sin(angle / 2)
    else:
        axis = np.asarray(axis)
        angle = np.asarray(angle)
        axis_n = normalize(axis)
        q = np.empty((axis.shape[0], 4), dtype=axis.dtype)
        q[:, 0] = np.cos(angle / 2)
        q[:, 1:] = axis_n * np.sin(angle / 2)

    return q


def as_axis_angle(q):
    # get the axis-angle representation from quaternion
    # q: (batch_size, 4)
    q = normalize(q)
    if _is_torch(q):
        angle = 2 * torch.acos(q[:, 0:1])
        axis = q[:, 1:] / torch.sin(angle / 2)
    else:
        angle = 2 * np.arccos(q[:, 0:1])
        axis = q[:, 1:] / np.sin(angle / 2)
    return axis, angle


def from_matrix(R):
    # get the quaternion from rotation matrix
    # R: (batch_size, 3, 3)
    if _is_torch(R):
        q = torch.empty(R.shape[0], 4, device=R.device, dtype=R.dtype)
        q[:, 0] = 0.5 * torch.sqrt(
            1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        )
        q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * q[:, 0])
        q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * q[:, 0])
        q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * q[:, 0])
    else:
        R = np.asarray(R)
        q = np.empty((R.shape[0], 4), dtype=R.dtype)
        q[:, 0] = 0.5 * np.sqrt(
            1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        )
        q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * q[:, 0])
        q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * q[:, 0])
        q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * q[:, 0])

    return q


def as_matrix(q):
    # get the rotation matrix from quaternion
    # q: (batch_size, 4)
    q = normalize(q)
    if _is_torch(q):
        R = torch.empty(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    else:
        R = np.empty((q.shape[0], 3, 3), dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)
    R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R[:, 1, 1] = 1 - 2 * (q[:, 1] ** 2 + q[:, 3] ** 2)
    R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    R[:, 2, 2] = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)

    return R


def mul(q1, q2):
    # q1: (batch_size, 4)
    # q2: (batch_size, 4)
    # return: q1 * q2: (batch_size, 4)
    if _is_torch(q1):
        q = torch.empty_like(q1)
    else:
        q = np.empty_like(q1)

    q[:, 0] = (
        q1[:, 0] * q2[:, 0]
        - q1[:, 1] * q2[:, 1]
        - q1[:, 2] * q2[:, 2]
        - q1[:, 3] * q2[:, 3]
    )
    q[:, 1] = (
        q1[:, 0] * q2[:, 1]
        + q1[:, 1] * q2[:, 0]
        + q1[:, 2] * q2[:, 3]
        - q1[:, 3] * q2[:, 2]
    )
    q[:, 2] = (
        q1[:, 0] * q2[:, 2]
        - q1[:, 1] * q2[:, 3]
        + q1[:, 2] * q2[:, 0]
        + q1[:, 3] * q2[:, 1]
    )
    q[:, 3] = (
        q1[:, 0] * q2[:, 3]
        + q1[:, 1] * q2[:, 2]
        - q1[:, 2] * q2[:, 1]
        + q1[:, 3] * q2[:, 0]
    )

    return q


def apply(q, a):
    # q: (batch_size, 4)
    # a: (batch_size, 3)
    # return: q * a * q^{-1}: (batch_size, 3)
    q = normalize(q)
    q_inv = conjugate(q)

    if _is_torch(q):
        zeros = torch.zeros(q.shape[0], 1, device=q.device, dtype=q.dtype)
        qa = torch.cat([zeros, a], dim=1)
    else:
        a = np.asarray(a)
        zeros = np.zeros((q.shape[0], 1), dtype=a.dtype)
        qa = np.concatenate([zeros, a], axis=1)

    return mul(mul(q, qa), q_inv)[:, 1:]