import numpy as np
import pytest

import thork as tk


def _bf16_dtype():
    """
    Return the numpy dtype to use for bfloat16 round-trips with ML Numpy
    builds (ml_dtypes) or a plain uint16 view otherwise. We compare in
    float32 so the storage representation doesn't matter for the test.
    """
    try:
        from ml_dtypes import bfloat16
        return np.dtype(bfloat16)
    except ImportError:
        return None


def _f32_to_bf16_u16(x : np.ndarray) -> np.ndarray:
    """
    Convert a float32 array to the bf16 bit pattern stored as uint16,
    using round-to-nearest-even truncation of the float32 bits.
    """
    u32 = x.view(np.uint32)
    rounding_bias = ((u32 >> 16) & 1) + 0x7FFF
    u16 = ((u32 + rounding_bias) >> 16).astype(np.uint16)
    return u16


def _bf16_u16_to_f32(u16 : np.ndarray) -> np.ndarray:
    """
    Convert a bf16 bit pattern stored as uint16 back to float32.
    """
    u32 = (u16.astype(np.uint32) << 16)
    return u32.view(np.float32)


def test_kittens_matmul_bf16():
    """
    Level-04-style tile-based matmul using TK register + shared tiles.

    Single warp per output tile (32 threads), 32x32 BLOCK_SIZE. Result is
    bf16; compare in float32 with a generous tolerance.
    """
    BLOCK = 32
    M = N = K = 128
    rng = np.random.default_rng(0)
    A_f32 = rng.standard_normal((M, K)).astype(np.float32) * 0.1
    B_f32 = rng.standard_normal((K, N)).astype(np.float32) * 0.1
    C_f32_expected = A_f32 @ B_f32

    A_bf = _f32_to_bf16_u16(A_f32)
    B_bf = _f32_to_bf16_u16(B_f32)
    C_bf = np.zeros((M, N), dtype=np.uint16)

    # NOTE: the numpy arrays are uint16; the kernel sees them as bf16.
    # We type-pun the buffer at the device boundary — thork's
    # kittens.Global[bf16] accepts uint16 storage transparently.

    @tk.jit
    def matmul(
        A   : tk.kittens.Global[tk.dt.bfloat16],
        B   : tk.kittens.Global[tk.dt.bfloat16],
        C   : tk.kittens.Global[tk.dt.bfloat16],
        N   : tk.Uint,
        bid : tk.Uint2[tk.BlockIdx],
    ):
        row = bid.y
        col = bid.x

        As = tk.kittens.st_bf(BLOCK, BLOCK)
        Bs = tk.kittens.st_bf(BLOCK, BLOCK)

        A_reg     = tk.kittens.rt_bf(BLOCK, BLOCK)
        B_reg     = tk.kittens.rt_bf(BLOCK, BLOCK)
        B_reg_col = tk.kittens.rt_bf_col(BLOCK, BLOCK)
        C_accum   = tk.kittens.rt_fl(BLOCK, BLOCK)

        tk.kittens.zero(C_accum)
        num_tiles = N // BLOCK
        for t in tk.range(num_tiles):
            tk.kittens.load(As, A, (row, t))
            tk.kittens.load(Bs, B, (t, col))
            tk.syncthreads()
            tk.kittens.load(A_reg, As)
            tk.kittens.load(B_reg, Bs)
            tk.kittens.swap_layout(B_reg_col, B_reg)
            tk.syncthreads()
            tk.kittens.mma_AB(C_accum, A_reg, B_reg_col, C_accum)
            tk.syncthreads()

        tk.kittens.store(C, C_accum, (row, col))

    matmul[
        (N // BLOCK, M // BLOCK, 1),
        (32, 1, 1),
    ](A_bf, B_bf, C_bf, N)

    C_f32 = _bf16_u16_to_f32(C_bf)
    np.testing.assert_allclose(C_f32, C_f32_expected, atol=2e-1, rtol=2e-2)


def test_kittens_tma_matmul_bf16():
    """
    Port of TK's educational level_05 matmul: TMA-based async loads into
    swizzled shared tiles, warp-level mma_AB, TMA store. Validates the
    full Global[dtype, (R, C)] + tma::* path end-to-end.
    """
    TILE_M = 64
    TILE_N = 64
    TILE_K = 32
    N = 128

    rng = np.random.default_rng(1)
    A_f32 = rng.standard_normal((N, N)).astype(np.float32) * 0.1
    B_f32 = rng.standard_normal((N, N)).astype(np.float32) * 0.1
    C_expected = A_f32 @ B_f32

    A_bf = _f32_to_bf16_u16(A_f32)
    B_bf = _f32_to_bf16_u16(B_f32)
    C_bf = np.zeros((N, N), dtype=np.uint16)

    @tk.jit
    def matmul(
        A   : tk.kittens.Global[tk.dt.bfloat16, (TILE_M, TILE_K)],
        B   : tk.kittens.Global[tk.dt.bfloat16, (TILE_K, TILE_N)],
        C   : tk.kittens.Global[tk.dt.bfloat16, (TILE_M, TILE_N)],
        N   : tk.Uint,
        bid : tk.Uint2[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
    ):
        col = bid.x
        row = bid.y

        As = tk.kittens.st_bf(TILE_M, TILE_K)
        Bs = tk.kittens.st_bf(TILE_K, TILE_N)
        Cs = tk.kittens.st_bf(TILE_M, TILE_N)

        smem_arrived = tk.kittens.semaphore()
        with tk.if_(tid == 0):
            tk.kittens.init_semaphore(smem_arrived, 0, 1)
        tk.syncthreads()

        A_reg     = tk.kittens.rt_bf(TILE_M, TILE_K)
        B_reg     = tk.kittens.rt_bf(TILE_K, TILE_N)
        B_reg_col = tk.kittens.rt_bf_col(TILE_K, TILE_N)
        C_accum   = tk.kittens.rt_fl(TILE_M, TILE_N)
        tk.kittens.zero(C_accum)

        num_tiles = N // TILE_K
        phase = tk.local(tk.dt.int32, 0)

        for t in tk.range(num_tiles):
            with tk.if_(tid == 0):
                a_bytes = TILE_M * TILE_K * 2
                b_bytes = TILE_K * TILE_N * 2
                tk.kittens.tma.expect_bytes(smem_arrived, a_bytes + b_bytes)
                tk.kittens.tma.load_async(As, A, (row, t), smem_arrived)
                tk.kittens.tma.load_async(Bs, B, (t, col), smem_arrived)

            tk.kittens.wait(smem_arrived, phase)
            phase ^= 1

            tk.kittens.load(A_reg, As)
            tk.kittens.load(B_reg, Bs)
            tk.kittens.swap_layout(B_reg_col, B_reg)
            tk.kittens.mma_AB(C_accum, A_reg, B_reg_col, C_accum)
            tk.syncthreads()

        tk.kittens.store(Cs, C_accum)
        tk.syncthreads()

        with tk.if_(tid == 0):
            tk.kittens.tma.store_async(C, Cs, (row, col))
            tk.kittens.tma.store_async_read_wait()

    matmul[
        (N // TILE_N, N // TILE_M, 1),
        (32, 1, 1),
    ](A_bf, B_bf, C_bf, N)

    C_f32 = _bf16_u16_to_f32(C_bf)
    np.testing.assert_allclose(C_f32, C_expected, atol=2e-1, rtol=2e-2)


def test_kittens_tcgen05_matmul_bf16_compiles_for_sm100():
    """
    Compile-only check of TK's educational level_06 matmul: TMA loads via
    semaphore, tcgen05 ``mm_AB`` / ``mma_AB`` into a Blackwell tensor-memory
    accumulator, ``warpgroup::load_async`` back to registers, then TMA
    store. The tcgen05 path is gated to ``KITTENS_SM10X`` upstream — TK
    doesn't expose it on consumer Blackwell (SM_120 / GB10), so we
    cross-compile against ``sm_100`` and skip the launch.
    """
    TILE_M = 128
    TILE_N = 128
    TILE_K = 64
    NUM_WARPS = 4

    @tk.jit
    def matmul(
        A   : tk.kittens.Global[tk.dt.bfloat16, (TILE_M, TILE_K)],
        B   : tk.kittens.Global[tk.dt.bfloat16, (TILE_K, TILE_N)],
        D   : tk.kittens.Global[tk.dt.bfloat16, (TILE_M, TILE_N)],
        N   : tk.Uint,
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
    ):
        wg_laneid = tk.kittens.warpgroup.laneid()

        grid_n = N // TILE_N
        bid_m  = bid // grid_n
        bid_n  = bid % grid_n

        a_smem = tk.kittens.st_bf(TILE_M, TILE_K)
        b_smem = tk.kittens.st_bf(TILE_K, TILE_N)
        d_smem = tk.kittens.st_bf(TILE_M, TILE_N)

        inputs_arrived  = tk.kittens.semaphore()
        inputs_finished = tk.kittens.semaphore()
        compute_done    = tk.kittens.semaphore()

        with tk.if_(tid == 0):
            tk.kittens.init_semaphore(inputs_arrived, 0, 1)
            tk.kittens.init_semaphore(inputs_finished, 1, 0)
            tk.kittens.init_semaphore(compute_done, 0, 1)
        tk.syncthreads()

        tm_alloc = tk.kittens.tensor_allocator()
        accum    = tk.kittens.tt_fl(TILE_M, TILE_N)
        with tk.if_(wg_laneid == 0):
            tk.kittens.tensor_alloc(accum, tm_alloc, 0)
        tk.kittens.warpgroup.sync(1)

        num_k = N // TILE_K
        phase = tk.local(tk.dt.int32, 0)

        for it in tk.range(num_k):
            with tk.if_(tid == 0):
                tk.kittens.wait(inputs_finished, phase ^ 1)
                tk.kittens.tma.expect_bytes(
                    inputs_arrived,
                    TILE_M * TILE_K * 2 + TILE_K * TILE_N * 2,
                )
                tk.kittens.tma.load_async(a_smem, A, (bid_m, it), inputs_arrived)
                tk.kittens.tma.load_async(b_smem, B, (it, bid_n), inputs_arrived)

            tk.kittens.wait(inputs_arrived, phase)
            phase ^= 1

            with tk.if_(wg_laneid == 0):
                with tk.if_(it == 0) as branch:
                    tk.kittens.warpgroup.mm_AB(accum, a_smem, b_smem, inputs_finished)
                with branch.else_():
                    tk.kittens.warpgroup.mma_AB(accum, a_smem, b_smem, inputs_finished)

        with tk.if_(wg_laneid == 0):
            tk.kittens.tcgen05_commit(compute_done, ncta=1)
        tk.kittens.wait(compute_done, 0)

        d_reg = tk.kittens.rt_bf(TILE_M // NUM_WARPS, TILE_N)
        tk.kittens.warpgroup.load_async(d_reg, accum)
        tk.kittens.tensor_load_wait()

        tk.kittens.warpgroup.sync(1)
        tk.kittens.warpgroup.store(d_smem, d_reg)
        tk.kittens.warpgroup.sync(1)

        with tk.if_(wg_laneid == 0):
            tk.kittens.tma.store_async(D, d_smem, (bid_m, bid_n))
            tk.kittens.tma.store_async_read_wait()

    matmul.compile_for_arch("100a")


def test_kittens_source_routes_through_nvcc():
    """
    Sanity check: the generated source for a TK kernel includes the
    kittens.cuh header and the shared-allocator preamble.
    """
    @tk.jit
    def kernel(
        A : tk.kittens.Global[tk.dt.bfloat16],
        B : tk.kittens.Global[tk.dt.bfloat16],
    ):
        As = tk.kittens.st_bf(32, 32)
        tk.kittens.load(As, A, (0, 0))
        tk.kittens.store(B, As, (0, 0))

    src = kernel.cuda_source
    assert '#include "kittens.cuh"' in src
    assert "__thork_smem_al" in src
    assert "kittens::gl<kittens::bf16, 1, 1, -1, -1>" in src
    assert "kittens::warp::load" in src
    assert "kittens::warp::store" in src
