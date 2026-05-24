"""
thork.verified — bridge between thork kernels and the stile-verifier type system.

Optional subpackage: requires ``stile-verifier`` to be installed. Provides
``@tvk.jit(spec=..., inputs=..., out_shape=..., out_dtype=...)`` — a
decorator that abstract-interprets a thork-style kernel body against
stile types, verifies the value written to the output pointer matches the
declared spec, then delegates the actual compile + launch to ``@tk.jit``.

Mirrors ``stile.triton.jit`` in spirit, adapted to thork's primitives:

- loads are ``ptr[i]`` (PointerTracer.__getitem__), not ``tl.load``;
- stores are ``out[i] = value`` (PointerTracer.__setitem__), not ``tl.store``;
- math intrinsics are ``tk.exp`` / ``tk.sqrt`` / etc., not ``tl.exp``;
- loops are ``for i in tk.range(...)``;
- attribute params (``Uint[BlockIdx]`` etc.) supply the thread-index
  symbols that drive the per-store slice extraction.
"""

try:
    import stile  # noqa: F401
    _HAS_STILE = True
except ImportError:
    _HAS_STILE = False


def _require_stile():
    if not _HAS_STILE:
        raise ImportError(
            "thork.verified requires the stile-verifier package. Install "
            "with `pip install stile-verifier`."
        )


if _HAS_STILE:
    from ._core import (
        Tensor,
        TypedThorkKernel,
        at,
        jit,
        load,
        range,
        store,
    )

    __all__ = [
        "Tensor",
        "TypedThorkKernel",
        "at",
        "jit",
        "load",
        "range",
        "store",
    ]
else:
    def __getattr__(name : str):
        _require_stile()

    __all__ = []
