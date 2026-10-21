"""Exercise real Moondream call sites with CPU storage and a recorded backend."""
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.moondream import attention_policy, text


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_does_not_require_the_cuda_extension(monkeypatch, device):
    monkeypatch.setattr(attention_policy, "get_runtime", lambda *_: pytest.fail("CUDA backend loaded"))
    attention_policy.validate_attention_policy(torch.device(device))


def test_old_kernel_fails_before_model_load(monkeypatch):
    def old_attention(q, k, v, **kwargs):
        pass
    monkeypatch.setattr(attention_policy, "get_runtime", lambda *_: SimpleNamespace(
        attention=SimpleNamespace(flash_attn_fwd=old_attention)))
    with pytest.raises(RuntimeError, match="companion kernel update"):
        attention_policy.validate_attention_policy(torch.device("cuda"))


def test_companion_kernel_is_accepted(monkeypatch):
    def new_attention(q, k, v, *, allow_persistent_split=True):
        pass
    monkeypatch.setattr(attention_policy, "get_runtime", lambda *_: SimpleNamespace(
        attention=SimpleNamespace(flash_attn_fwd=new_attention)))
    attention_policy.validate_attention_policy(torch.device("cuda"))


@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("cuda", [False, True])
def test_real_attention_call_disables_selector_without_retry(monkeypatch, mode, cuda):
    calls = []
    def attention(q, k, v, **kwargs):
        calls.append(kwargs)
        return torch.ones_like(q), None
    backend = SimpleNamespace(attention=SimpleNamespace(flash_attn_fwd=attention))
    monkeypatch.setattr(text, "get_runtime", lambda *_: backend)
    monkeypatch.setattr(text, "rotary_embedding", lambda *_: None)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: cuda))
    module = SimpleNamespace(qkv=SimpleNamespace(weight=torch.zeros(12, 4, dtype=torch.bfloat16), bias=None),
                             proj=SimpleNamespace(weight=torch.eye(4, dtype=torch.bfloat16), bias=None))
    monkeypatch.setattr(text, "_kestrel_linear", lambda x, w, b=None: torch.nn.functional.linear(x, w, b))
    cache = SimpleNamespace(cache=SimpleNamespace(k_cache=torch.zeros(2, 1, 1, 4),
                                                  v_cache=torch.zeros(2, 1, 1, 4)),
                            update=lambda *a, **k: None)
    result = text.attn(torch.zeros(1, 1, 4, dtype=torch.bfloat16), module,
        torch.empty(0), cache, None, 1, 1, torch.zeros(1, 1, dtype=torch.long),
        mode=mode, slot_mapping=torch.zeros(1, dtype=torch.long), page_table=torch.zeros(1, 1, dtype=torch.int32),
        paged_kv_seqlens_q=torch.ones(1, dtype=torch.int32),
        paged_kv_seqlens_k=torch.ones(1, dtype=torch.int32))
    assert len(calls) == 1
    assert (calls[0].get("allow_persistent_split") is False) if cuda else ("allow_persistent_split" not in calls[0])
    assert result.shape == (1, 1, 4)
