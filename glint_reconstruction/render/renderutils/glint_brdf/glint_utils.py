import torch
import numpy as np

EPSILON = 1e-6


@torch.no_grad()
def print_gpu_mem():
    # Check if a GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()

        # Loop over each GPU
        for i in range(num_gpus):
            # Get GPU name
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

            # Get current memory usage
            memory_allocated = torch.cuda.memory_allocated(i)
            max_memory_allocated = torch.cuda.max_memory_allocated(i)
            memory_cached = torch.cuda.memory_cached(i)
            max_memory_cached = torch.cuda.max_memory_cached(i)

            print(f"Memory Allocated: {memory_allocated / (1024**2):.2f} MB")
            print(f"Max Memory Allocated: {max_memory_allocated / (1024**2):.2f} MB")
            print(f"Memory Cached: {memory_cached / (1024**2):.2f} MB")
            print(f"Max Memory Cached: {max_memory_cached / (1024**2):.2f} MB \n")

        print("--------------------------------------------------------------")
    else:
        print("No GPU available.")


@torch.no_grad()
def is_valid(tensor):
    valid = True
    if torch.any(torch.isnan(tensor)):
        print(f"The input tensor has NaN values.")
        valid = False
    if torch.any(torch.isinf(tensor)):
        print(f"The input tensor has Inf values.")
        valid = False

    return valid


def remove_zeros(tensor):
    res = torch.where(tensor == 0.0, tensor.mean(), tensor)
    if torch.any(res == 0.0):
        res = torch.where(res == 0.0, EPSILON, res)

    return res


def toIntApprox(tensor):
    return torch.where(tensor >= 0.0, torch.floor(tensor), torch.ceil(tensor))


def normalise_to(tensor, min, max):
    min_val = tensor.min()
    max_val = tensor.max()
    return min + ((tensor - min_val) * (max - min)) / remove_zeros(max_val - min_val)


def scale(tensor, min=0.0, max=1.0):
    assert min <= max
    return (max - min) * tensor + min


"""
The codes below are modified from Glints2023.hlsl in https://drive.google.com/file/d/1YQDxlkZFEwV6ZeaXCUYMhB4P-3ODS32e/view.
Credit to Deliot et al. (https://thomasdeliot.wixsite.com/blog/single-post/hpg23-real-time-rendering-of-glinty-appearance-using-distributed-binomial-laws-on-anisotropic-grids).
"""


@torch.no_grad()
def pcg3dFloat(tensor):
    v = np.uint32(tensor) * np.uint32(1664525) * np.uint32(1013904223)

    v[..., 0] += v[..., 1] * v[..., 2]
    v[..., 1] += v[..., 2] * v[..., 0]
    v[..., 2] += v[..., 0] * v[..., 1]

    v ^= v >> np.uint32(16)

    v[..., 0] += v[..., 1] * v[..., 2]
    v[..., 1] += v[..., 2] * v[..., 0]
    v[..., 2] += v[..., 0] * v[..., 1]

    return torch.tensor(v * (1.0 / 4294967296.0), dtype=torch.float32, device="cuda")


@torch.no_grad()
def asuint(f):
    return np.array(f, dtype=np.float32).view(np.uint32)


@torch.no_grad()
def f16tof32(val):
    tmp = np.array([val & 0xFFFF], dtype=np.uint32).view(np.float16)
    return tmp.astype(np.float32)


@torch.no_grad()
def UnpackFloatParallel4(input: np.ndarray):
    uintInput = asuint(input.astype(np.float32))
    a = f16tof32(uintInput >> 16)
    b = f16tof32(uintInput)

    a = torch.tensor(np.squeeze(a)[..., ::2], dtype=torch.float32).view(4, -1, 3, 4)
    b = torch.tensor(np.squeeze(b)[..., ::2], dtype=torch.float32).view(4, -1, 3, 4)

    res = torch.cat((a, b)).view(2, 4, -1, 3, 4).to("cuda")

    assert is_valid(res)
    assert res[0].min() >= 0.0 and res[0].max() <= 1.0

    return res


def sampleNormalDistribution(u, mu, sigma):
    # 2.0 * u - 1.0 must be within (-1.0, 1.0) => u must be in (0.0, 1.0)
    # shape of (sigma * 1.414213).unsqueeze(-1)[None].repeat(4,1,1) is (4, n, 1)
    res = (sigma * 1.414213).unsqueeze(-1)[None].repeat(4, 1, 1) * torch.erfinv(
        torch.clamp(2.0 * u - 1.0, min=-0.9999999, max=0.9999999)
    ) + mu.unsqueeze(-1)[None].repeat(
        4, 1, 1
    )  # (4, n, 3)
    assert torch.all(torch.isfinite(res))
    return res


def HashWithoutSine13(p3):
    p3 = torch.frac(p3 * 0.1031)
    p3 += torch.sum(p3 * (p3[..., [1, 2, 0]] + 33.33), dim=-1).unsqueeze(-1)

    res = torch.frac((p3[..., 0] + p3[..., 1]) * p3[..., 2]).unsqueeze(-1)

    return res


def GetGradientEllipse(duvdx, duvdy):
    # Construct Jacobian matrix
    # Note that HLSL is column major: https://stackoverflow.com/questions/22756121/confuse-with-row-major-and-column-major-matrix-multiplication-in-hlsl#:~:text=HLSL%20uses%20Column%2DMajor%20and%20XNAMath%20uses%20ROW%2DMajor.
    J = torch.transpose(torch.stack([duvdx, duvdy], dim=2), 1, 2)
    # a random tensor act as a replacement for the zero determinant matrices
    replacement = torch.tensor([[0.6533, 0.4048], [0.8707, 0.2704]], device="cuda")
    # Check if determinant is zero and replace the matrices
    J = torch.where(
        torch.det(J).unsqueeze(1).unsqueeze(2) == 0,
        replacement,
        J,
    )

    J = torch.linalg.inv(J)
    J = torch.matmul(J, torch.transpose(J, 1, 2))

    a = J[..., 0, 0]
    c = J[..., 1, 0]
    d = J[..., 1, 1]

    T = a + d
    D = torch.linalg.det(J)
    # They are meant to be > 0.0
    L1 = remove_zeros(torch.abs(T / 2.0 - torch.pow(T * T / 3.99999 - D, 0.5)))
    L2 = remove_zeros(torch.abs(T / 2.0 + torch.pow(T * T / 3.99999 - D, 0.5)))

    A0 = torch.stack((L1 - d, c), dim=-1)
    A1 = torch.stack((L2 - d, c), dim=-1)
    r0 = 1.0 / torch.sqrt(L1)
    r1 = 1.0 / torch.sqrt(L2)

    ellipseMajor = torch.nn.functional.normalize(A0, dim=-1) * r0.unsqueeze(-1)
    ellipseMinor = torch.nn.functional.normalize(A1, dim=-1) * r1.unsqueeze(-1)

    assert is_valid(ellipseMajor) and is_valid(ellipseMinor)

    return ellipseMajor, ellipseMinor


def VectorToSlope(v):
    res = torch.tensor([-v[0] / v[2], -v[1] / v[2]])
    assert is_valid(res)
    return res


def SlopeToVector(s):
    z = 1 / remove_zeros(
        torch.sqrt(torch.clamp(s[0] * s[0] + s[1] * s[1] + 1, min=0.0))
    )
    x = s[0] * z
    y = s[1] * z

    res = torch.tensor([x, y, z])

    return res


def RotateUV(uv, rotation, mid):
    a = (
        torch.cos(rotation) * (uv[..., 0].unsqueeze(-1) - mid[0])
        + torch.sin(rotation) * (uv[..., 1].unsqueeze(-1) - mid[1])
        + mid[0]
    )
    b = (
        torch.cos(rotation) * (uv[..., 1].unsqueeze(-1) - mid[1])
        - torch.sin(rotation) * (uv[..., 0].unsqueeze(-1) - mid[0])
        + mid[1]
    )
    return torch.cat((a, b), dim=-1)


def BilinearLerp(values, valuesLerp):
    resultX = torch.lerp(values[..., 0], values[..., 2], valuesLerp[..., 0])
    resultY = torch.lerp(values[..., 1], values[..., 3], valuesLerp[..., 0])

    res = torch.lerp(resultX, resultY, valuesLerp[..., 1])

    return res


def Remap(s, a1, a2, b1, b2):
    return b1 + (s - a1) * (b2 - b1) / remove_zeros(a2 - a1)


def Remap01To(s, b1, b2):
    return b1 + s * (b2 - b1)


def RemapTo01(s, a1, a2):
    return (s - a1) / remove_zeros(a2 - a1)


def GetBarycentricWeightsTetrahedron(p, v):
    v_diff = v[:3, ...] - v[-1, ...]
    c11 = v_diff[0]
    c21 = v_diff[1]
    c31 = v_diff[2]
    c41 = v[3] - p

    m1 = c31[..., 1:] / remove_zeros(c31[..., 0]).unsqueeze(-1)
    c12 = c11[..., 1:] - c11[..., 0].unsqueeze(-1) * m1
    c22 = c21[..., 1:] - c21[..., 0].unsqueeze(-1) * m1
    c32 = c41[..., 1:] - c41[..., 0].unsqueeze(-1) * m1

    m2 = c22[..., 1] / remove_zeros(c22[..., 0])
    uvwk_0 = (c32[..., 0] * m2 - c32[..., 1]) / remove_zeros(
        c12[..., 1] - c12[..., 0] * m2
    )
    uvwk_1 = -(c32[..., 0] + c12[..., 0] * uvwk_0) / remove_zeros(c22[..., 0])
    uvwk_2 = -(
        c41[..., 0] + c21[..., 0] * uvwk_1 + c11[..., 0] * uvwk_0
    ) / remove_zeros(c31[..., 0])
    uvwk_3 = 1.0 - uvwk_2 - uvwk_1 - uvwk_0

    res = torch.nn.functional.softmax(
        torch.stack((uvwk_0, uvwk_1, uvwk_2, uvwk_3), dim=-1), dim=-1
    )

    # The range of the return values depends on the input vertices and the position of the point p relative to the
    # tetrahedron. In general, the barycentric coordinates should satisfy the condition 0 <= uvwk.x, uvwk.y, uvwk.z, uvwk.w <= 1
    # and uvwk.x + uvwk.y + uvwk.z + uvwk.w = 1. These conditions ensure that the point p lies within the tetrahedron.
    return res
