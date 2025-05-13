import torch

def norm(q):
    # q: (batch_size, 4)

    return torch.sqrt(torch.sum(q**2, dim=1, keepdim=True))

def normalize(q):
    # q: (batch_size, 4)

    return q / (norm(q) + 1e-20)

def conjugate(q):
    # q: (batch_size, 4)

    return torch.cat([q[:, 0:1], -q[:, 1:4]], dim=1)

def inverse(q):
    # q: (batch_size, 4)

    return conjugate(q) / (torch.sum(q**2, dim=1, keepdim=True) + 1e-20)


def from_vectors(a, b):
    # get the quaternion from two 3D vectors, such that b = qa.
    # a: (batch_size, 3)
    # b: (batch_size, 3)
    # note: a and b don't need to be unit vectors.

    q = torch.empty(a.shape[0], 4, device=a.device)
    q[:, 0] = torch.sqrt(torch.sum(a**2, dim=1)) * torch.sqrt(torch.sum(b**2, dim=1)) + torch.sum(a * b, dim=1)
    q[:, 1:] = torch.cross(a, b)
    q = normalize(q)

    return q

def from_axis_angle(axis, angle):
    # get the quaternion from axis-angle representation
    # axis: (batch_size, 3)
    # angle: (batch_size, 1), in radians

    q = torch.empty(axis.shape[0], 4, device=axis.device)
    q[:, 0] = torch.cos(angle / 2)
    q[:, 1:] = normalize(axis) * torch.sin(angle / 2)
    
    return q

def as_axis_angle(q):
    # get the axis-angle representation from quaternion
    # q: (batch_size, 4)

    q = normalize(q)
    angle = 2 * torch.acos(q[:, 0:1])
    axis = q[:, 1:] / torch.sin(angle / 2)

    return axis, angle

def from_matrix(R):
    # get the quaternion from rotation matrix
    # R: (batch_size, 3, 3)

    q = torch.empty(R.shape[0], 4, device=R.device)
    q[:, 0] = 0.5 * torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * q[:, 0])
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * q[:, 0])
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * q[:, 0])

    return q

def as_matrix(q):
    # get the rotation matrix from quaternion
    # q: (batch_size, 4)

    R = torch.empty(q.shape[0], 3, 3, device=q.device)
    R[:, 0, 0] = 1 - 2 * (q[:, 2]**2 + q[:, 3]**2)
    R[:, 0, 1] = 2 * (q[:, 1] * q[:, 2] - q[:, 0] * q[:, 3])
    R[:, 0, 2] = 2 * (q[:, 1] * q[:, 3] + q[:, 0] * q[:, 2])
    R[:, 1, 0] = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R[:, 1, 1] = 1 - 2 * (q[:, 1]**2 + q[:, 3]**2)
    R[:, 1, 2] = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R[:, 2, 0] = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    R[:, 2, 1] = 2 * (q[:, 2] * q[:, 3] + q[:, 0] * q[:, 1])
    R[:, 2, 2] = 1 - 2 * (q[:, 1]**2 + q[:, 2]**2)

    return R

def mul(q1, q2):
    # q1: (batch_size, 4)
    # q2: (batch_size, 4)
    # return: q1 * q2: (batch_size, 4)

    q = torch.empty_like(q1)
    q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    return q

def apply(q, a):
    # q: (batch_size, 4)
    # a: (batch_size, 3)
    # return: q * a * q^{-1}: (batch_size, 3)

    q = normalize(q)
    q_inv = conjugate(q)

    return mul(mul(q, torch.cat([torch.zeros(q.shape[0], 1, device=q.device), a], dim=1)), q_inv)[:, 1:]