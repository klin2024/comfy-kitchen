# SPDX-FileCopyrightText: Copyright (c) 2026 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import get_capable_backends

FP8 = torch.float8_e4m3fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ref(x, w, b=None):
    """What the eager path (and ComfyUI's manual cast) computes."""
    return functional.linear(x, w.to(x.dtype), b.to(x.dtype) if b is not None else None)


def _ref32(x, w, b=None):
    """fp32 reference on the same operands (bias rounded to x's dtype, as the reference does)."""
    out = x.float() @ w.float().t()
    if b is not None:
        out = out + b.to(x.dtype).float()
    return out


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30)).item()


def _bias(kind, n, device):
    if kind == "none":
        return None
    b = torch.randn(n, device=device) * 0.02
    return b.to({"fp8": FP8, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[kind])


def _inputs(m, k, n, dtype, bias="fp8", device="cuda"):
    """x ~ N(0, 1); the weight in the range of DiT checkpoints, stored as FP8."""
    x = torch.randn(m, k, device=device).to(dtype)
    w = (torch.randn(n, k, device=device) * 0.02).to(FP8)
    return x, w, _bias(bias, n, device)


def _require_triton():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    if "triton" not in get_capable_backends("linear_w8a16", "cuda"):
        pytest.skip("triton backend not capable for linear_w8a16")


def _triton(x, w, b=None):
    kwargs = {"x": x, "weight": w, "bias": b}
    valid = ck.registry.validate_backend_for_call("triton", "linear_w8a16", kwargs)
    if not valid.success:
        pytest.skip(f"triton refuses this call here: {valid.failed_param}: {valid.failure_reason}")
    with ck.use_backend("triton"):
        return ck.linear_w8a16(x, w, b)


def _check(out, x, w, b):
    """Within bf16/fp16 rounding of the reference: the operands are identical, only the
    fp32 summation order differs."""
    ref = _ref(x, w, b)
    assert out.shape == ref.shape
    assert out.dtype == x.dtype
    assert _rel(out, ref) <= 2e-2
    assert _rel(out, _ref32(x, w, b)) <= 1e-2


# (M, K, N): Qwen-Image / Flux.1 linears (M 1 modulation, M 5 / 46 / 262 text,
# K 64 img_in, N 64 proj_out) at reduced M, plus M and N that are not tile multiples.
_SHAPES = [
    (1, 256, 3072),
    (1, 3072, 6144),
    (3, 512, 1000),
    (5, 3584, 3072),
    (8, 1024, 4100),
    (9, 512, 3000),
    (46, 3072, 3072),
    (262, 1024, 12288),
    (300, 12288, 320),
    (513, 3072, 64),
    (1000, 64, 3072),
]


# ---------------------------------------------------------------------------
# Triton
# ---------------------------------------------------------------------------


class TestLinearW8A16Triton:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("shape", _SHAPES)
    def test_matches_reference(self, shape, dtype, seed):
        _require_triton()
        m, k, n = shape
        x, w, b = _inputs(m, k, n, dtype, "fp8")
        _check(_triton(x, w, b), x, w, b)

    @pytest.mark.parametrize("bias", ["none", "fp8", "bf16", "fp16", "fp32"])
    @pytest.mark.parametrize("m", [4, 200])
    def test_bias_dtypes(self, bias, m, seed):
        """Any float bias is rounded to x's dtype first, as the reference does."""
        _require_triton()
        x, w, b = _inputs(m, 512, 384, torch.bfloat16, bias)
        _check(_triton(x, w, b), x, w, b)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("m", [1, 8, 128])
    def test_every_fp8_value_exact(self, dtype, m):
        """x = rows of the identity: out[i, j] = weight[j, i], exactly. Covers every
        finite e4m3fn value (incl. subnormals and +-0) on the GEMV and GEMM paths."""
        _require_triton()
        codes = torch.arange(256, dtype=torch.int32, device="cuda")
        codes = codes[(codes & 0x7F) != 0x7F]  # 254 finite values; 0x7f / 0xff are NaN
        k, n = 128, codes.numel()
        idx = (torch.arange(n, device="cuda")[:, None] + torch.arange(k, device="cuda")) % n
        w = codes[idx].to(torch.uint8).view(FP8)
        x = torch.eye(k, device="cuda", dtype=dtype)[:m]
        out = _triton(x, w)
        assert torch.equal(out, w.t()[:m].to(dtype))

    def test_leading_dims_and_strided_rows(self, seed):
        """(B, S, K) input whose rows are a column slice of a wider buffer, and a weight
        whose rows are a slice of a wider buffer: both read in place."""
        _require_triton()
        buf = torch.randn(2, 37, 1024, device="cuda", dtype=torch.bfloat16)
        x = buf[..., :512]
        wbuf = (torch.randn(640, 1024, device="cuda") * 0.02).to(FP8)
        w = wbuf[:, 256:768]
        b = _bias("fp8", 640, "cuda")
        out = _triton(x, w, b)
        assert out.shape == (2, 37, 640)
        _check(out, x, w, b)
        x1 = buf[0, :3, :512]
        _check(_triton(x1, w, b), x1, w, b)

    def test_zero_rows(self):
        _require_triton()
        x = torch.empty(3, 0, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.zeros(96, 256, device="cuda").to(FP8)
        out = _triton(x, w)
        assert out.shape == (3, 0, 96)
        assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("path", ["small_m", "large_m"])
    def test_every_config_covers_output(self, path, seed):
        """Each autotune config on its own writes every output element correctly at a
        shape no tile divides (M and N masked). Configs that do not fit this GPU's
        shared memory are skipped, as the autotuner skips them."""
        _require_triton()
        from triton.runtime.errors import OutOfResources

        from comfy_kitchen.backends.triton import w8a16 as mod

        tuner = mod._small_m_kernel if path == "small_m" else mod._large_m_kernel
        m = 5 if path == "small_m" else 77
        x, w, b = _inputs(m, 512, 200, torch.bfloat16, "fp8")
        configs = list(tuner.configs)
        ran = 0
        try:
            for cfg in configs:
                tuner.configs = [cfg]
                tuner.cache.clear()
                try:
                    out = _triton(x, w, b)
                except OutOfResources:
                    continue
                ran += 1
                assert _rel(out, _ref(x, w, b)) <= 2e-2, str(cfg)
        finally:
            tuner.configs = configs
            tuner.cache.clear()
        assert ran > 0

    def test_dispatch_selects_triton(self, seed):
        _require_triton()
        x, w, b = _inputs(16, 256, 128, torch.bfloat16)
        kwargs = {"x": x, "weight": w, "bias": b}
        assert ck.registry.get_capable_backend("linear_w8a16", kwargs) == "triton"


# ---------------------------------------------------------------------------
# Eager reference and fallback
# ---------------------------------------------------------------------------


class TestLinearW8A16Eager:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("bias", ["none", "fp8", "fp32"])
    def test_eager_is_reference(self, dtype, bias, device, seed):
        if device == "cpu" and dtype == torch.float16:
            pytest.skip("fp16 linear on CPU not checked")
        x, w, b = _inputs(6, 64, 48, dtype, bias, device)
        x = x.reshape(2, 3, 64)
        with ck.use_backend("eager"):
            out = ck.linear_w8a16(x, w, b)
        assert out.shape == (2, 3, 48)
        assert torch.equal(out, _ref(x, w, b))

    @pytest.mark.parametrize(
        "case", ["fp32_x", "e5m2_weight", "k_not_multiple", "weight_col_stride", "cpu"]
    )
    def test_fallback(self, case, seed):
        """Calls the triton kernel does not serve are refused there and run the eager
        reference instead."""
        if not torch.cuda.is_available() or "triton" not in get_capable_backends(
            "linear_w8a16", "cuda"
        ):
            pytest.skip("triton backend required")
        device = "cpu" if case == "cpu" else "cuda"
        dtype = torch.float32 if case in ("fp32_x", "cpu") else torch.bfloat16
        k = 100 if case == "k_not_multiple" else 256
        x, w, b = _inputs(10, k, 64, dtype, "fp8", device)
        if case == "e5m2_weight":
            w = w.to(torch.bfloat16).to(torch.float8_e5m2)
        elif case == "weight_col_stride":
            w = w.t().contiguous().t()
        kwargs = {"x": x, "weight": w, "bias": b}
        assert not ck.registry.validate_backend_for_call("triton", "linear_w8a16", kwargs).success
        out = ck.linear_w8a16(x, w, b)
        assert torch.equal(out, _ref(x, w, b))

    def test_fake_shape(self):
        x = torch.empty(2, 7, 96, device="meta", dtype=torch.bfloat16)
        w = torch.empty(40, 96, device="meta", dtype=FP8)
        out = torch.ops.comfy_kitchen.linear_w8a16(x, w, None)
        assert out.shape == (2, 7, 40)
        assert out.dtype == torch.bfloat16
