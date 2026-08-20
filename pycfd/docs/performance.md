# Performance

Where the wall-clock actually goes, measured rather than assumed.

---

Measured on an 8-thread machine. The stencil kernel is JIT-compiled in two
builds — serial and thread-parallel — and selected by grid size, because
threading only pays for itself above roughly 12k cells.

| grid | NumPy stencil | Numba stencil | speed-up |
|---|---:|---:|---:|
| 64×64 | 0.123 ms | 0.055 ms | 2.25× |
| 128×128 | 0.415 ms | 0.223 ms | 1.86× |
| 192×192 | 1.112 ms | 0.505 ms | 2.20× |
| 256×256 | 2.110 ms | 0.359 ms | **5.87×** |

Full solver, wall-clock seconds per 1000 time steps:

| grid | NumPy | Numba | speed-up |
|---|---:|---:|---:|
| 128×128 | 7.46 s | 6.99 s | 1.07× |
| 256×256 | 33.6 s | 27.6 s | 1.22× |

**The end-to-end gain is small, and that is the honest headline.** Profiling
shows the sparse LU solve dominates a time step; the stencils are not the
bottleneck, so even a large speed-up there moves the total little. The task
specification's target of ≥3× applies to the stencil operations and is met at
256×256 but not at 128×128, where thread-overhead limits the gain to ~1.9×.

The Numba path is *bit-identical* to the NumPy path — same operations in the
same order, verified in the test suite for both blending modes and both boundary
families, and the serial and parallel builds agree bitwise with each other.

> If you modify `kernels.py`: the parallel build must not set `cache=True`.
> Numba keys its on-disk cache on the source function, so two builds of the same
> function collide and the second silently loads the first one's artifact —
> yielding a "parallel" kernel that runs serially. That cost a real 2.5× before
> it was caught.

---

---

[← Back to the README](../README.md)
