# Building and Testing PaddlePaddle on AMD Instinct GPUs with ROCm

This branch carries the AMD ROCm and HIP support for PaddlePaddle v3.4.0. It
builds Paddle from source for AMD Instinct GPUs and produces a Python wheel.
There are no prebuilt packages for this branch.

## What You Need

- An AMD Instinct GPU. The steps were run on an AMD Instinct MI355X (`gfx950`).
  AMD Instinct MI300X uses `gfx942`.
- Ubuntu 24.04 with Python 3.12.
- ROCm 10.1.0. Earlier ROCm releases do not build this branch.
- About 30 GB of free disk space and a many-core build host. A full build
  compiles several thousand translation units.

## 1. Install the Build Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake patch patchelf wget curl \
  python3 python3-dev python3-venv
```

## 2. Install ROCm

Install ROCm from the public ROCm Python packages into a virtual environment.
Use `device-gfx942` instead of `device-gfx950` for AMD Instinct MI300X.

```bash
python3 -m venv paddle-rocm
. paddle-rocm/bin/activate
pip install --upgrade pip
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "rocm[libraries,devel,device-gfx950]==10.1.0"

export ROCM_PATH=$(rocm-sdk path --root)
export HIP_PATH=$ROCM_PATH
export PATH=$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib/llvm/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}

rocminfo | grep "Marketing Name"
```

Keep these environment variables set for the build and whenever you import
Paddle. Paddle loads the ROCm libraries and the OpenMP runtime from
`LD_LIBRARY_PATH`, and the import fails without it.

## 3. Get the Source

```bash
git clone --depth 1 --branch rocm-enablement https://github.com/AMD-Ecosystem/Paddle.git
cd Paddle
```

## 4. Apply the rocPRIM Host Guards

Paddle host C++ code includes rocPRIM headers through hipCUB and rocThrust. Two
constructs in the ROCm 10.1 rocPRIM headers only compile in device code. The
helper script adds a guard for each. It edits two headers inside the ROCm
installation at `$ROCM_PATH`, and it is safe to run more than once.

```bash
bash tools/rocm/apply_rocprim_host_guards.sh
```

## 5. Install the Python Build Dependencies

```bash
pip install -r python/requirements.txt \
  "protobuf<=3.20.2" "numpy<2" "pybind11-stubgen<3" \
  pyyaml requests decorator astor opt_einsum setuptools wheel
```

The three version limits matter. Paddle v3.4.0 generates code against the
protobuf 3.20 API, builds against the NumPy 1.x C API, and calls the
`pybind11-stubgen` 2.x interface when it packages the wheel.

## 6. Configure and Build

Set `ARCH` to the feature-qualified target ID of your GPU. Read it from
`rocminfo`.

```bash
rocminfo | grep "amdhsa--gfx"
```

For AMD Instinct MI350X and MI355X the value is `gfx950:sramecc+:xnack-`. For
AMD Instinct MI300X it is `gfx942:sramecc+:xnack-`. A bare name such as
`gfx950` builds, and then every GPU kernel fails to load at run time.

```bash
ARCH="gfx950:sramecc+:xnack-"

for attempt in 1 2 3 4 5; do
  cmake -S . -B build \
    -DWITH_ROCM=ON -DWITH_GPU=OFF -DWITH_RCCL=ON \
    -DWITH_TESTING=OFF -DWITH_DISTRIBUTE=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$ROCM_PATH/lib/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="$ROCM_PATH/lib/llvm/bin/clang++" \
    -DROCM_PATH="$ROCM_PATH" -DHIP_PATH="$ROCM_PATH" \
    -DCMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
    -DPY_VERSION=3.12 -DPYTHON_EXECUTABLE="$(which python)" \
    -DCMAKE_HIP_ARCHITECTURES="$ARCH" -DAMDGPU_TARGETS="$ARCH" -DGPU_ARCHS="$ARCH" \
    && break
  echo "configure attempt $attempt failed, retrying"
done

cmake --build build -j"$(nproc)"
```

The configure step downloads several third-party archives, and a download can
fail partway. The loop runs the configure step again when that happens.

The wheel is written to `build/python/dist/`.

## 7. Install and Test

```bash
pip install --force-reinstall --no-deps build/python/dist/paddlepaddle_dcu-*.whl

cd /tmp
python "$OLDPWD/tools/rocm/verify_gpu.py"
```

Run the check from outside the source tree so that Python imports the
installed wheel. The script confirms that the build has ROCm support, runs
the built-in Paddle installation check, compares nine operators between the
CPU and the GPU, and runs a short training loop on the GPU. It prints
`RESULT: PASS` when everything succeeds.

You can also check a build by hand.

```python
import paddle

print(paddle.device.is_compiled_with_rocm())  # True
paddle.set_device("gpu:0")
x = paddle.randn([512, 512])
print(float((x @ x).abs().mean()))
```

## What the ROCm Support Changes

- CMake support for the ROCm 10.1 toolchain, including the requested GPU
  architecture list and the vendored third-party patches.
- Host C++ code compiles as plain C++. `thrust/complex.h` is included only in
  device code.
- CUDA-only dynamic-loader headers are guarded so that a ROCm build does not
  include them.
- Sorting and top-k kernels use the rocPRIM and hipCUB interfaces in ROCm
  10.1 and handle the 64-lane wavefront width of AMD Instinct GPUs.

## Not Included

- BF16 convolution through MIOpen.
- FlashAttention. The `extern_flashattn` dependency is disabled for ROCm
  builds.
