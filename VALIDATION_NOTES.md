# Building and validating PaddlePaddle on ROCm (AMD Instinct)

This branch refreshes the AMD ROCm/HIP support in Paddle to upstream v3.4.0 and
targets the ROCm 10 build floor on AMD Instinct GPUs (gfx942 base; gfx950
primary). The AMD divergence versus upstream v3.4.0 is a thin overlay (13 files:
build glue under `cmake/`, a few ROCm backend helpers under `paddle/phi/`, and
three vendored-dependency patches under `patches/`).

## Toolchain / environment

- ROCm 10.0 or newer. Validated on the manylinux base image
  `registry.aifoundry.amd.com/rocm-manylinux228-rhel810:10.0.0a20260730`
  (`/opt/rocm/.info/version` = 10.0.0, HIP internal 7.15.x, clang 23.0.0).
- AMD Instinct MI300X (gfx942) confirmed via `rocminfo` inside the build
  container. gfx950 is the primary target and must also be built/validated.

## REQUIRED: compile the C/C++ side with the ROCm clang

On ROCm 6+, rocThrust bundles CCCL/libcudacxx. `thrust/complex.h` (included by
`paddle/phi/common/complex.h`) pulls in `cuda/std/cmath` fp16/bf16 math overloads
that reference intrinsics (`__hisnan`, `__hisinf`, `__habs`, `__hmax`,
`__double2bfloat16`) only available under Clang's HIP language mode
(`defined(__clang__) && defined(__HIP__)`). A plain host g++ compile of any
translation unit that transitively includes thrust (for example
`paddle/pir/src/core/block_argument.cc`) fails with `'... has not been declared'`,
and a non-HIP Clang C++ compile fails on host-only `__half`/`__hip_bfloat16`
conversion ambiguities in the CCCL headers. `__HIP_NO_HALF_CONVERSIONS__` /
`__HIP_NO_HALF_OPERATORS__` do NOT resolve this.

The overlay `cmake/hip.cmake` now adds `-x hip -fno-gpu-rdc` to the host C++
flags, but only when the C++ compiler is Clang. So the ROCm build MUST be
configured with the ROCm clang as the C/C++ compiler:

```
-DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang
-DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
```

(Use the actual `${ROCM_PATH}/llvm/bin` on the manylinux base.) On a non-Clang
compiler the `-x hip` guard no-ops, which keeps a CPU/GCC configure safe but does
NOT fix the thrust host-compile break -- ROCm builds must use clang.

## Configure

Env seams on the manylinux base (see the hipSHIFT porter/validator notes):

```
git config --global --add safe.directory '*'         # source tree bind-mount ownership
dnf install -y wget python3.11-devel                 # jit download test + shared libpython
export MIOPEN_USER_DB_PATH=/work/miopen MIOPEN_CUSTOM_CACHE_DIR=/work/miopen
```

The manylinux `cp3xx` interpreters are static-only (no shared libpython); use the
RHEL8 system python3.11 which ships `/usr/lib64/libpython3.11.so.1.0`.

```
cmake -S . -B build \
  -DWITH_ROCM=ON -DWITH_RCCL=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \        # gfx950 for the primary target
  -DPYTHON_EXECUTABLE=/usr/bin/python3.11 \
  -DPYTHON_INCLUDE_DIR=/usr/include/python3.11 \
  -DPYTHON_LIBRARY=/usr/lib64/libpython3.11.so.1.0 -DPY_VERSION=3.11
```

The overlay auto-sets `CMAKE_HIP_ARCHITECTURES` from the arch list, so passing it
explicitly is belt-and-braces (and is how the gfx950 target selects its arch).

## Build

```
cmake --build build -j$(nproc)
```

Note: `-x hip` compiles the host C++ side in HIP language mode, which increases
build time versus a plain g++ build (device codegen passes run for host TUs).
This is inherent to the ROCm 10 CCCL bundle and unavoidable for any host TU that
includes thrust.

## Validate

Run Paddle's documented test suite on the built wheel (gfx942 hard gate, gfx950
primary). At minimum: `python -c "import paddle; paddle.utils.run_check()"` plus
small conv/matmul/gelu forward+backward, then the unit-test suite for the touched
backends. Record PASS/FAIL/SKIP per arch plus coverage.

## Known human-track gaps (not ported here)

- BF16 convolution via MIOpen (Immediate/solution API hand-tuning).
- FlashAttention on ROCm (routes through AOTriton / Composable Kernel, not this
  source path); `extern_flashattn` is guarded off on all ROCm >= 6 builds.
