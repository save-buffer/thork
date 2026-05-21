import numpy as np

import thork as tk


def test_matrix_add():
    shape = (1024, 1024)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    expected = A + B

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

    matrix_add[
        (int(np.prod(shape)) // 128, 1, 1),
        (128, 1, 1),
    ](
        C,
        A,
        B,
    )

    np.testing.assert_allclose(C, expected)


def test_matmul_simple():
    M = N = K = 256

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    expected = A @ B

    @tk.jit
    def matmul_simple(
        out : tk.DevicePointer[tk.dt.float32],
        A   : tk.DevicePointer[tk.dt.float32],
        B   : tk.DevicePointer[tk.dt.float32],
        K   : tk.Uint,
        N   : tk.Uint,
        bid : tk.Uint2[tk.BlockIdx],
        tid : tk.Uint2[tk.ThreadIdx],
        bdm : tk.Uint2[tk.BlockDim],
    ):
        row = bid.y * bdm.y + tid.y
        col = bid.x * bdm.x + tid.x

        acc = tk.local(tk.dt.float32, 0.0)
        for k in tk.range(K):
            acc += A[row * K + k] * B[k * N + col]

        out[row * N + col] = acc

    matmul_simple[
        (N // 16, M // 16, 1),
        (16, 16, 1),
    ](
        C,
        A,
        B,
        K,
        N,
    )

    np.testing.assert_allclose(C, expected, atol=1e-3, rtol=1e-3)


def test_matmul_tiled():
    M = N = K = 128
    TILE = 16

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    expected = A @ B

    @tk.jit
    def matmul_tiled(
        out : tk.DevicePointer[tk.dt.float32],
        A   : tk.DevicePointer[tk.dt.float32],
        B   : tk.DevicePointer[tk.dt.float32],
        M   : tk.Uint,
        N   : tk.Uint,
        K   : tk.Uint,
        bid : tk.Uint2[tk.BlockIdx],
        tid : tk.Uint2[tk.ThreadIdx],
    ):
        A_tile = tk.shared(tk.dt.float32, (TILE, TILE))
        B_tile = tk.shared(tk.dt.float32, (TILE, TILE))

        row = bid.y * TILE + tid.y
        col = bid.x * TILE + tid.x

        acc = tk.local(tk.dt.float32, 0.0)
        for k_tile in tk.range(0, K, TILE):
            A_tile[tid.y, tid.x] = A[row * K + (k_tile + tid.x)]
            B_tile[tid.y, tid.x] = B[(k_tile + tid.y) * N + col]
            tk.syncthreads()

            for k in tk.range(TILE):
                acc += A_tile[tid.y, k] * B_tile[k, tid.x]
            tk.syncthreads()

        out[row * N + col] = acc

    matmul_tiled[
        (N // TILE, M // TILE, 1),
        (TILE, TILE, 1),
    ](
        C,
        A,
        B,
        M,
        N,
        K,
    )

    np.testing.assert_allclose(C, expected, atol=1e-3, rtol=1e-3)


def test_sigmoid():
    """
    Exercises math intrinsics (tk.exp) by computing 1 / (1 + exp(-x)).
    """
    shape = (4096,)
    A = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    expected = 1.0 / (1.0 + np.exp(-A))

    @tk.jit
    def sigmoid(
        out : tk.DevicePointer[tk.dt.float32],
        A   : tk.DevicePointer[tk.dt.float32],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        out[i] = 1.0 / (1.0 + tk.exp(-A[i]))

    sigmoid[
        (shape[0] // 128, 1, 1),
        (128, 1, 1),
    ](C, A)

    np.testing.assert_allclose(C, expected, atol=1e-4, rtol=1e-4)


def test_histogram():
    """
    Each thread reads one value from `values`, computes its bin, and bumps
    the bin counter with tk.atomic_add.
    """
    N = 4096
    NBINS = 16

    np.random.seed(0)
    values = np.random.randint(0, NBINS * 4, size=N).astype(np.uint32)
    counts = np.zeros(NBINS, dtype=np.uint32)

    expected = np.bincount(values % NBINS, minlength=NBINS).astype(np.uint32)

    @tk.jit
    def histogram(
        counts : tk.DevicePointer[tk.dt.uint32],
        values : tk.DevicePointer[tk.dt.uint32],
        nbins  : tk.Uint,
        bid    : tk.Uint[tk.BlockIdx],
        tid    : tk.Uint[tk.ThreadIdx],
        bdm    : tk.Uint[tk.BlockDim],
    ):
        gid = bid * bdm + tid
        val = values[gid]
        bin_idx = val % nbins
        tk.atomic_add(counts, bin_idx, 1)

    histogram[
        (N // 128, 1, 1),
        (128, 1, 1),
    ](counts, values, NBINS)

    np.testing.assert_array_equal(counts, expected)


def test_warp_sum_reduction():
    """
    One warp per output element; each lane handles a strided slice of K,
    then warp_sum reduces to a single value lane 0 writes out.
    """
    M = N = 64
    K = 128

    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    expected = A @ B

    @tk.jit
    def matmul_warp(
        out  : tk.DevicePointer[tk.dt.float32],
        A    : tk.DevicePointer[tk.dt.float32],
        B    : tk.DevicePointer[tk.dt.float32],
        K    : tk.Uint,
        N    : tk.Uint,
        bid  : tk.Uint2[tk.BlockIdx],
        lane : tk.Uint[tk.LaneId],
        wsz  : tk.Uint[tk.WarpSize],
    ):
        row = bid.y
        col = bid.x

        partial = tk.local(tk.dt.float32, 0.0)
        for k in tk.range(lane, K, wsz):
            partial += A[row * K + k] * B[k * N + col]

        total = tk.warp_sum(partial)
        with tk.if_(lane == 0):
            out[row * N + col] = total

    matmul_warp[
        (N, M, 1),
        (32, 1, 1),
    ](
        C,
        A,
        B,
        K,
        N,
    )

    np.testing.assert_allclose(C, expected, atol=1e-3, rtol=1e-3)


def test_device_fn_and_control_flow():
    """
    Exercises @tk.device_fn (newton_sqrt), tk.while_, tk.break_, and
    tk.if_().else_() in one kernel.
    """
    N = 4096
    A = np.random.rand(N).astype(np.float32) * 100.0
    A[::128] = -A[::128]
    C = np.zeros(N, dtype=np.float32)

    expected = np.where(A < 0.0, 0.0, np.sqrt(np.abs(A))).astype(np.float32)

    @tk.device_fn
    def newton_sqrt(x : tk.dt.float32) -> tk.dt.float32:
        guess = tk.local(tk.dt.float32, x * 0.5)
        i = tk.local(tk.dt.uint32, 0)
        with tk.while_(i < 20):
            new_guess = 0.5 * (guess + x / guess)
            with tk.if_(tk.fabs(new_guess - guess) < 1e-6):
                tk.break_()
            guess.assign(new_guess)
            i += 1
        return guess

    @tk.jit
    def sqrt_kernel(
        out : tk.DevicePointer[tk.dt.float32],
        A   : tk.DevicePointer[tk.dt.float32],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = A[i]
        with tk.if_(x < 0.0) as branch:
            out[i] = 0.0
        with branch.else_():
            out[i] = newton_sqrt(x)

    sqrt_kernel[(N // 128, 1, 1), (128, 1, 1)](C, A)

    np.testing.assert_allclose(C, expected, atol=1e-4, rtol=1e-4)


def test_source_map_captures_user_locs():
    """
    Verify the JittedKernel's source_map points generated-CUDA lines back at
    the originating Python source locations.
    """
    THIS_FILE = __file__

    @tk.jit
    def add_two(
        out : tk.DevicePointer[tk.dt.float32],
        A   : tk.DevicePointer[tk.dt.float32],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = A[i] + 1.0
        out[i] = x + 1.0  # this line should appear in source_map

    smap = add_two.source_map
    assert smap is not None and len(smap) > 0, "source_map should be populated"
    files = {loc[0] for loc in smap.values()}
    assert files == {THIS_FILE}, (
        f"Expected all source_map entries to point at {THIS_FILE}, got {files}"
    )


def test_kernels_library():
    """
    The pre-built thork.kernels.* functions should drop in and match
    numpy on representative inputs.
    """
    shape = (512, 512)
    A = np.random.randn(*shape).astype(np.float32)
    B = np.random.randn(*shape).astype(np.float32)
    C = np.zeros(shape, dtype=np.float32)

    add = tk.kernels.matrix_add(shape)
    add(C, A, B)
    np.testing.assert_allclose(C, A + B)

    M = N = K = 128
    A2 = np.random.randn(M, K).astype(np.float32)
    B2 = np.random.randn(K, N).astype(np.float32)
    C2 = np.zeros((M, N), dtype=np.float32)
    mm = tk.kernels.matmul(M, N, K)
    mm(C2, A2, B2)
    np.testing.assert_allclose(C2, A2 @ B2, atol=1e-3, rtol=1e-3)
