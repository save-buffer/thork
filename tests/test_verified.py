import numpy as np
import pytest


pytest.importorskip("stile")

import stile
import thork as tk
import thork.verified as tvk
from stile import dim


@pytest.fixture(autouse=True)
def _reset():
    with stile.scope():
        yield


def test_scalar_multiply_verification():
    """
    Verification half: typed-load + arithmetic + typed-store accepted
    against spec ``2 * X:N``. Shapes live on the parameter annotations.
    """
    N = dim("VN", 128)

    @tvk.jit(spec="2 * X:VN")
    def times_two(
        out : tvk.Tensor[tk.dt.float32, "VN"],
        X   : tvk.Tensor[tk.dt.float32, "X:VN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        tvk.store(out, x * 2, N[i:i + 1])

    assert isinstance(times_two, tvk.TypedThorkKernel)


def test_scalar_multiply_runs():
    """
    Full path: verify + dispatch the kernel and check numerics on the GPU.
    """
    N = dim("VN", 128)
    block = 32

    @tvk.jit(spec="2 * X:VN")
    def times_two(
        out : tvk.Tensor[tk.dt.float32, "VN"],
        X   : tvk.Tensor[tk.dt.float32, "X:VN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        tvk.store(out, x * 2, N[i:i + 1])

    X = np.random.randn(N.size).astype(np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    times_two[(N.size // block, 1, 1), (block, 1, 1)](out, X)
    np.testing.assert_allclose(out, X * 2)


def test_vector_add():
    """
    Two-input typed loads: ``out[i] = X[i] + Y[i]`` against ``X + Y``.
    """
    N = dim("VAN", 256)
    block = 32

    @tvk.jit(spec="X:VAN + Y:VAN")
    def vadd(
        out : tvk.Tensor[tk.dt.float32, "VAN"],
        X   : tvk.Tensor[tk.dt.float32, "X:VAN"],
        Y   : tvk.Tensor[tk.dt.float32, "Y:VAN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        y = tvk.load(Y, N[i:i + 1])
        tvk.store(out, x + y, N[i:i + 1])

    X = np.random.randn(N.size).astype(np.float32)
    Y = np.random.randn(N.size).astype(np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    vadd[(N.size // block, 1, 1), (block, 1, 1)](out, X, Y)
    np.testing.assert_allclose(out, X + Y, rtol=1e-5, atol=1e-5)


def test_exp_intrinsic():
    """
    ``tk.exp`` flows through ``tvk.load`` results and maps to
    ``stile.exp`` in the verifier.
    """
    N = dim("EN", 128)
    block = 32

    @tvk.jit(spec="exp(X:EN)")
    def kexp(
        out : tvk.Tensor[tk.dt.float32, "EN"],
        X   : tvk.Tensor[tk.dt.float32, "X:EN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        tvk.store(out, tk.exp(x), N[i:i + 1])

    X = (np.random.randn(N.size) * 0.5).astype(np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    kexp[(N.size // block, 1, 1), (block, 1, 1)](out, X)
    np.testing.assert_allclose(out, np.exp(X), rtol=1e-4, atol=1e-4)


def test_2d_indexing_uses_static_stride():
    """
    Two-axis typed load on a 2-D input should compile to ``X[m * K_size + n]``
    with ``K_size`` baked in from the static dim shape.
    """
    M = dim("M2", 64)
    K = dim("K2", 32)

    @tvk.jit(spec="X:M2 K2")
    def identity_2d(
        out : tvk.Tensor[tk.dt.float32, "M2 K2"],
        X   : tvk.Tensor[tk.dt.float32, "X:M2 K2"],
        bid : tk.Uint2[tk.BlockIdx],
        tid : tk.Uint2[tk.ThreadIdx],
    ):
        m = bid.y
        n = tid.x
        val = tvk.load(X, M[m:m + 1], K[n:n + 1])
        tvk.store(out, val, M[m:m + 1], K[n:n + 1])

    # Assert the codegen baked the K dim size as a constant stride.
    src = identity_2d.cuda_source
    assert "* 32" in src, f"expected K_size=32 stride in source:\n{src}"

    X = np.random.randn(M.size, K.size).astype(np.float32).ravel()
    out = np.zeros(M.size * K.size, dtype=np.float32)
    identity_2d[(1, M.size, 1), (K.size, 1, 1)](out, X)
    np.testing.assert_array_equal(out, X)


def test_verification_rejects_wrong_spec():
    """
    Verification must reject when the tile's ExprType doesn't match
    the spec restricted to the same tile. Trace-based verification
    fires at first launch.
    """
    N = dim("WN", 64)
    block = 32

    @tvk.jit(spec="3 * X:WN")
    def wrong(
        out : tvk.Tensor[tk.dt.float32, "WN"],
        X   : tvk.Tensor[tk.dt.float32, "X:WN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        tvk.store(out, x * 2, N[i:i + 1])

    X = np.zeros(N.size, dtype=np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    with pytest.raises(ValueError, match="verification failed"):
        wrong[(N.size // block, 1, 1), (block, 1, 1)](out, X)


def test_verification_rejects_no_store():
    """
    Vacuous-pass guard: a kernel that never calls ``tvk.store`` is
    rejected at first launch.
    """
    N = dim("NSN", 32)
    block = 32

    @tvk.jit(spec="X:NSN")
    def noop(
        out : tvk.Tensor[tk.dt.float32, "NSN"],
        X   : tvk.Tensor[tk.dt.float32, "X:NSN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        _ = tvk.load(X, N[i:i + 1])

    X = np.zeros(N.size, dtype=np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    with pytest.raises(ValueError, match="never writes"):
        noop[(N.size // block, 1, 1), (block, 1, 1)](out, X)


def test_bare_subscript_load_rejected():
    """
    Bare ``X[i]`` produces a Tracer with no ``_stype``; the value
    derived from it can't be verified at ``tvk.store`` time. The
    decorator raises a "no recoverable stile type" error.
    """
    N = dim("BLN", 64)
    block = 32

    @tvk.jit(spec="2 * X:BLN")
    def bad(
        out : tvk.Tensor[tk.dt.float32, "BLN"],
        X   : tvk.Tensor[tk.dt.float32, "X:BLN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = X[i]                              # <-- bare load, no _stype
        tvk.store(out, x * 2, N[i:i + 1])

    X = np.zeros(N.size, dtype=np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    with pytest.raises(ValueError, match="recoverable stile type"):
        bad[(N.size // block, 1, 1), (block, 1, 1)](out, X)


def test_bare_subscript_store_rejected():
    """
    Bare ``out[i] = ...`` traces a store but doesn't go through
    ``tvk.store``, so the verifier's "stored" flag stays False — the
    decorator raises a "never writes" error.
    """
    N = dim("BSN", 64)
    block = 32

    @tvk.jit(spec="2 * X:BSN")
    def bad(
        out : tvk.Tensor[tk.dt.float32, "BSN"],
        X   : tvk.Tensor[tk.dt.float32, "X:BSN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        out[i] = x * 2                        # <-- bare store

    X = np.zeros(N.size, dtype=np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    with pytest.raises(ValueError, match="never writes"):
        bad[(N.size // block, 1, 1), (block, 1, 1)](out, X)


def test_matmul_verification():
    """
    A canonical thread-per-element matmul. The verifier should fold the
    ``for k in tvk.range(K):`` accumulator into a ``Reduce(sum, K, A*B)``
    and certify it against the einsum spec ``(A:Mv Kv, B:Kv Nv -> Mv Nv)``.
    Uses ``tvk.at(DIM, idx)`` for the scalar-slice sugar.
    """
    Mv = dim("Mv", 16)
    Nv = dim("Nv", 16)
    Kv = dim("Kv", 8)

    @tvk.jit(spec="(A:Mv Kv, B:Kv Nv -> Mv Nv)")
    def matmul(
        out : tvk.Tensor[tk.dt.float32, "Mv Nv"],
        A   : tvk.Tensor[tk.dt.float32, "A:Mv Kv"],
        B   : tvk.Tensor[tk.dt.float32, "B:Kv Nv"],
        bid : tk.Uint2[tk.BlockIdx],
    ):
        m = bid.y
        n = bid.x
        acc = tk.local(tk.dt.float32, 0.0)
        for k in tvk.range(Kv):
            a = tvk.load(A, tvk.at(Mv, m), tvk.at(Kv, k))
            b = tvk.load(B, tvk.at(Kv, k), tvk.at(Nv, n))
            acc += a * b
        tvk.store(out, acc, tvk.at(Mv, m), tvk.at(Nv, n))

    assert isinstance(matmul, tvk.TypedThorkKernel)


def test_matmul_runs():
    """
    End-to-end: verify the matmul, then dispatch and check numerics.
    """
    Mr = dim("Mr", 16)
    Nr = dim("Nr", 16)
    Kr = dim("Kr", 8)

    @tvk.jit(spec="(A:Mr Kr, B:Kr Nr -> Mr Nr)")
    def matmul(
        out : tvk.Tensor[tk.dt.float32, "Mr Nr"],
        A   : tvk.Tensor[tk.dt.float32, "A:Mr Kr"],
        B   : tvk.Tensor[tk.dt.float32, "B:Kr Nr"],
        bid : tk.Uint2[tk.BlockIdx],
    ):
        m = bid.y
        n = bid.x
        acc = tk.local(tk.dt.float32, 0.0)
        for k in tvk.range(Kr):
            a = tvk.load(A, tvk.at(Mr, m), tvk.at(Kr, k))
            b = tvk.load(B, tvk.at(Kr, k), tvk.at(Nr, n))
            acc += a * b
        tvk.store(out, acc, tvk.at(Mr, m), tvk.at(Nr, n))

    A = np.random.randn(Mr.size, Kr.size).astype(np.float32)
    B = np.random.randn(Kr.size, Nr.size).astype(np.float32)
    out = np.zeros(Mr.size * Nr.size, dtype=np.float32)
    matmul[(Nr.size, Mr.size, 1), (1, 1, 1)](out, A.ravel(), B.ravel())
    np.testing.assert_allclose(
        out.reshape(Mr.size, Nr.size), A @ B, rtol=1e-4, atol=1e-4,
    )


def test_at_and_slice_forms_interchangeable():
    """
    `tvk.at(DIM, idx)` and `DIM[idx:idx+1]` produce the same Sliced;
    a kernel using one verifies, and a kernel using the other compiles
    to identical CUDA source.
    """
    N = dim("AtN", 64)
    block = 32

    @tvk.jit(spec="2 * X:AtN")
    def via_at(
        out : tvk.Tensor[tk.dt.float32, "AtN"],
        X   : tvk.Tensor[tk.dt.float32, "X:AtN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, tvk.at(N, i))
        tvk.store(out, x * 2, tvk.at(N, i))

    @tvk.jit(spec="2 * X:AtN")
    def via_slice(
        out : tvk.Tensor[tk.dt.float32, "AtN"],
        X   : tvk.Tensor[tk.dt.float32, "X:AtN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        tvk.store(out, x * 2, N[i:i + 1])

    # Both forms generate the same per-thread element index `bid * bdm + tid`.
    assert via_at.cuda_source.replace(via_at.name, "") == \
           via_slice.cuda_source.replace(via_slice.name, "")


def test_matmul_rejects_wrong_accumulator():
    """
    If the kernel accumulates ``a + b`` instead of ``a * b``, the
    verifier should reject — the body's ExprType doesn't match the
    einsum's inner product. Trace-based: fires at first launch.
    """
    Mw = dim("Mw", 16)
    Nw = dim("Nw", 16)
    Kw = dim("Kw", 8)

    @tvk.jit(spec="(A:Mw Kw, B:Kw Nw -> Mw Nw)")
    def matmul_bad(
        out : tvk.Tensor[tk.dt.float32, "Mw Nw"],
        A   : tvk.Tensor[tk.dt.float32, "A:Mw Kw"],
        B   : tvk.Tensor[tk.dt.float32, "B:Kw Nw"],
        bid : tk.Uint2[tk.BlockIdx],
    ):
        m = bid.y
        n = bid.x
        acc = tk.local(tk.dt.float32, 0.0)
        for k in tvk.range(Kw):
            a = tvk.load(A, Mw[m:m + 1], Kw[k:k + 1])
            b = tvk.load(B, Kw[k:k + 1], Nw[n:n + 1])
            acc += a + b                                 # <-- wrong op
        tvk.store(out, acc, Mw[m:m + 1], Nw[n:n + 1])

    A = np.zeros((Mw.size, Kw.size), dtype=np.float32)
    B = np.zeros((Kw.size, Nw.size), dtype=np.float32)
    out = np.zeros(Mw.size * Nw.size, dtype=np.float32)
    with pytest.raises(ValueError, match="verification failed"):
        matmul_bad[(Nw.size, Mw.size, 1), (1, 1, 1)](out, A.ravel(), B.ravel())


def test_abs_intrinsic():
    """
    ``tk.abs`` flows through the verifier as ``stile.abs`` (which
    lowers to ``max(x, -x)``).
    """
    N = dim("AbsN", 128)
    block = 32

    @tvk.jit(spec="abs(X:AbsN)")
    def kabs(
        out : tvk.Tensor[tk.dt.float32, "AbsN"],
        X   : tvk.Tensor[tk.dt.float32, "X:AbsN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        tvk.store(out, tk.abs(x), N[i:i + 1])

    X = np.random.randn(N.size).astype(np.float32) * 2.0
    out = np.zeros(N.size, dtype=np.float32)
    kabs[(N.size // block, 1, 1), (block, 1, 1)](out, X)
    np.testing.assert_allclose(out, np.abs(X), rtol=1e-5, atol=1e-5)


def test_max_intrinsic_relu_style():
    """
    ``tk.max(x, 0)`` types as ``maximum(X, 0)`` — the natural
    relu-via-max form.
    """
    N = dim("MaxN", 128)
    block = 32

    @tvk.jit(spec="relu(X:MaxN)")
    def krelu(
        out : tvk.Tensor[tk.dt.float32, "MaxN"],
        X   : tvk.Tensor[tk.dt.float32, "X:MaxN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        tvk.store(out, tk.max(x, 0.0), N[i:i + 1])

    X = np.random.randn(N.size).astype(np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    krelu[(N.size // block, 1, 1), (block, 1, 1)](out, X)
    np.testing.assert_allclose(out, np.maximum(X, 0.0), rtol=1e-5, atol=1e-5)


def test_cast_preserves_stype():
    """
    ``tk.cast`` changes the dtype but not the logical stile expression,
    so verification still succeeds.
    """
    N = dim("CN", 128)
    block = 32

    @tvk.jit(spec="2 * X:CN")
    def kcast(
        out : tvk.Tensor[tk.dt.float32, "CN"],
        X   : tvk.Tensor[tk.dt.float32, "X:CN"],
        bid : tk.Uint[tk.BlockIdx],
        tid : tk.Uint[tk.ThreadIdx],
        bdm : tk.Uint[tk.BlockDim],
    ):
        i = bid * bdm + tid
        x = tvk.load(X, N[i:i + 1])
        # Round-trip through float -> float keeps the stype intact;
        # exercises the cast.stype-passthrough path.
        x_cast = tk.cast(x * 2, tk.dt.float32)
        tvk.store(out, x_cast, N[i:i + 1])

    X = np.random.randn(N.size).astype(np.float32)
    out = np.zeros(N.size, dtype=np.float32)
    kcast[(N.size // block, 1, 1), (block, 1, 1)](out, X)
    np.testing.assert_allclose(out, X * 2, rtol=1e-5, atol=1e-5)


def test_plain_devicepointer_rejected():
    """
    A pointer parameter annotated with the plain ``tk.DevicePointer``
    has no stile shape — the decorator must steer the user toward the
    Tensor annotation.
    """
    N = dim("PDN", 32)

    with pytest.raises(TypeError, match="tvk.Tensor"):
        @tvk.jit(spec="X:PDN")
        def bad(
            out : tvk.Tensor[tk.dt.float32, "PDN"],
            X   : tk.DevicePointer[tk.dt.float32],
            bid : tk.Uint[tk.BlockIdx],
            tid : tk.Uint[tk.ThreadIdx],
            bdm : tk.Uint[tk.BlockDim],
        ):
            i = bid * bdm + tid
            x = tvk.load(X, N[i:i + 1])
            tvk.store(out, x, N[i:i + 1])
