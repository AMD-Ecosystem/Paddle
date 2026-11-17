#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Applies two small host-compile guards to the rocPRIM headers of a ROCm 10.1
# installation. Paddle host C++ translation units include rocPRIM headers
# through hipCUB and rocThrust, and two constructs in those headers only
# compile in device code:
#
#   1. rocprim/intrinsics/arch.hpp: size() calls the device-only builtin
#      __builtin_amdgcn_wavefrontsize.
#   2. rocprim/detail/various.hpp: uses the device macro __align__.
#
# The script is idempotent. Run it once after installing ROCm and before
# configuring Paddle:
#
#   export ROCM_PATH=$(rocm-sdk path --root)
#   bash tools/rocm/apply_rocprim_host_guards.sh
set -euo pipefail

: "${ROCM_PATH:?Set ROCM_PATH to the ROCm installation prefix first}"
INC="${ROCM_PATH}/include"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH_HPP="${INC}/rocprim/intrinsics/arch.hpp"
VARIOUS_HPP="${INC}/rocprim/detail/various.hpp"

[ -f "${ARCH_HPP}" ] || { echo "error: ${ARCH_HPP} not found" >&2; exit 1; }
[ -f "${VARIOUS_HPP}" ] || { echo "error: ${VARIOUS_HPP} not found" >&2; exit 1; }

PATCH_FILE="${HERE}/rocprim_wavefront_size_host_guard.patch"
if patch -p3 -R -s -f --dry-run -d "${INC}" < "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "arch.hpp: wavefront size host guard already present"
else
  patch -p3 -s -d "${INC}" < "${PATCH_FILE}"
  echo "arch.hpp: wavefront size host guard applied"
fi

MARK="Paddle host guard for __align__"
if grep -q "${MARK}" "${VARIOUS_HPP}"; then
  echo "various.hpp: __align__ host guard already present"
else
  sed -i "1i // ${MARK}\n#ifndef __align__\n#define __align__(x) __attribute__((aligned(x)))\n#endif" "${VARIOUS_HPP}"
  echo "various.hpp: __align__ host guard applied"
fi
