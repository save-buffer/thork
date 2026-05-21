# Thork: A Tracing DSL for NVIDIA GPUs

Thork is a DSL for writing kernels for NVIDIA GPUs. You can think of it as a Python wrapper
on top of CUDA C++. This makes development more convenient, since it makes the kernels live
in the same source language as most ML code, and makes it easy to verify correctness against
NumPy.

Unlike Triton, Thork is a tracing DSL rather than a parsing-based DSL. This means that a Thork
program is fundamentally a Python program that creates a CUDA program. Thork can use any Python
libraries to aid in metaprogramming.

Thork is the sister project to [Spork](https://github.com/save-buffer/spork) (the same idea, but
for Apple GPUs and Metal).

## Installation

Thork runs on machines with an NVIDIA GPU and CUDA toolkit installed (nvcc + nvrtc). Install with:

```bash
uv add thork
```

or with pip:

```bash
pip install thork
```

```python
import thork as tk
```

## Example: Matrix Addition

In NumPy, you'd write:
```python
shape = (1024, 1024)
A = np.random.randn(*shape).astype(np.float32)
B = np.random.randn(*shape).astype(np.float32)
out = A + B
```

A CUDA kernel to do the same:
```cuda
extern "C" __global__ void matrix_add(float *out, const float *A, const float *B) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = A[i] + B[i];
}
```

The equivalent thork kernel:
```python
@tk.jit
def matrix_add(
    out : tk.DevicePointer[tk.dt.float32],
    A   : tk.DevicePointer[tk.dt.float32],
    B   : tk.DevicePointer[tk.dt.float32],
    bid : tk.Uint[tk.BlockIdx],
    tid : tk.Uint[tk.ThreadIdx],
    bdm : tk.Uint[tk.BlockDim],
):
    i = bid * bdm + tid
    out[i] = A[i] + B[i]
```

The attribute parameters (`tk.Uint[tk.BlockIdx]`, etc.) expand at the top of the generated
kernel body into the matching CUDA built-in (`blockIdx.x`, `threadIdx.x`, `blockDim.x`).
For vector versions (`tk.Uint3[tk.BlockIdx]`) you get a value with `.x`, `.y`, `.z` fields.

To launch:
```python
matrix_add[
    (int(np.prod(shape)) // 128, 1, 1),
    (128, 1, 1),
](
    C,
    A,
    B,
)
```

The first bracketed tuple is the grid; the second is the block. In the parentheses you pass
the NumPy arrays — Thork allocates device memory, copies inputs, launches, and copies back
any array that the kernel wrote to.
