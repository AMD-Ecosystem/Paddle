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
"""Functional check for a ROCm build of Paddle on an AMD GPU.

Compares a handful of operators between the CPU and the GPU, checks that GPU
outputs are not all zero, and runs a short training loop on the GPU. Exits
non-zero on any failure.
"""

import sys

import numpy as np

import paddle
import paddle.nn.functional as F

TOL = 1e-3


def main():
    print("paddle", paddle.__version__)
    if not paddle.device.is_compiled_with_rocm():
        print("FAIL: this Paddle build was not compiled with ROCm")
        return 1
    count = paddle.device.cuda.device_count()
    print("GPU count:", count)
    if count < 1:
        print("FAIL: no GPU visible")
        return 1
    paddle.utils.run_check()

    data = np.random.default_rng(0).standard_normal((256, 256)).astype("float32")
    ops = [
        ("relu", F.relu),
        ("add", lambda x: x + x),
        ("multiply", lambda x: x * 3.0),
        ("matmul", lambda x: x @ x),
        ("softmax", lambda x: F.softmax(x, axis=-1)),
        ("sum", lambda x: x.sum(axis=0)),
        ("topk", lambda x: paddle.topk(x, k=5)[0]),
        ("argsort", lambda x: paddle.argsort(x, axis=-1).astype("float32")),
        ("gelu", F.gelu),
    ]
    failed = False
    for name, fn in ops:
        paddle.set_device("cpu")
        cpu = fn(paddle.to_tensor(data)).numpy()
        paddle.set_device("gpu:0")
        gpu = fn(paddle.to_tensor(data)).numpy()
        diff = float(np.abs(cpu - gpu).max())
        ok = diff <= TOL * max(1.0, float(np.abs(cpu).max())) and bool(np.any(gpu != 0))
        print(f"{name:9s} max abs diff {diff:.3e}  {'PASS' if ok else 'FAIL'}")
        failed |= not ok

    paddle.set_device("gpu:0")
    paddle.seed(0)
    layer = paddle.nn.Linear(256, 10)
    opt = paddle.optimizer.SGD(parameters=layer.parameters(), learning_rate=0.1)
    x = paddle.to_tensor(data)
    y = paddle.randint(0, 10, [256])
    first = last = None
    for _ in range(20):
        loss = F.cross_entropy(layer(x), y)
        loss.backward()
        opt.step()
        opt.clear_grad()
        last = float(loss)
        first = last if first is None else first
    ok = last < first
    print(f"training  loss {first:.4f} -> {last:.4f}  {'PASS' if ok else 'FAIL'}")
    failed |= not ok

    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
