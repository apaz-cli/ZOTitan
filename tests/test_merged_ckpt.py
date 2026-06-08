"""Roundtrip test: wrapped LoRA model -> save_pretrained(merged_state_dict) -> from_pretrained.

The merged checkpoint, loaded as a plain HF model, must reproduce the wrapped
model's logits (the merge is mathematically what LoRALinear.forward computes).
Run on GPU 1: CUDA_VISIBLE_DEVICES=1 .venv/bin/python tests/test_merged_ckpt.py
"""
import tempfile
import torch
from transformers import AutoModelForCausalLM

from zotitan.model import load_model, ModelConfig
from zotitan.lora import make_lora, LoRAConfig, merged_state_dict


def main():
    model, tokenizer = load_model(ModelConfig())
    make_lora(model, LoRAConfig())

    # Make adapters non-trivial so the delta actually matters.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.02)

    model.eval()
    batch = tokenizer(["The quick brown fox jumps over the lazy dog."],
                      return_tensors="pt").to(model.device)

    def logits(m):
        with torch.no_grad():
            return m(**batch).logits.float().cpu()

    def roundtrip(dtype):
        with tempfile.TemporaryDirectory() as d:
            sd = merged_state_dict(model)
            assert all(v.device.type == "cpu" for v in sd.values()), "state dict not on CPU"
            assert not any(".lora_A." in k or ".lora_B." in k or ".base_linear." in k
                           for k in sd), "wrapper keys leaked into state dict"
            model.save_pretrained(d, state_dict=sd)
            reloaded = AutoModelForCausalLM.from_pretrained(
                d, dtype=dtype, device_map="cuda").eval()
            return logits(reloaded)

    # One fp32 reference from the live wrapped model. Convert once and never flip
    # back — LoRALinear caches lora_dtype, so re-casting the live model mid-test
    # would desync adapter dtype from the cached cast (a test artifact, not a bug
    # in the checkpoint path, which never re-casts the model).
    model.float()
    ref = logits(model)

    # Correctness gate: in fp32 the merge is exact by construction.
    got32 = roundtrip(torch.float32)
    max32 = (ref - got32).abs().max().item()
    agree32 = torch.argmax(ref, -1).eq(torch.argmax(got32, -1)).float().mean().item()
    print(f"[fp32] max|logit diff| = {max32:.4e}   argmax agreement = {agree32:.4f}")
    assert max32 < 1e-2, f"fp32 merge not exact: {max32}"
    assert agree32 == 1.0, f"fp32 argmax mismatch: {agree32}"

    # Informational: production loads bf16, so the saved (fp32) weights get
    # downcast on load. This quantifies pure storage-dtype cost vs the fp32 ref.
    got16 = roundtrip(torch.bfloat16)
    max16 = (ref - got16).abs().max().item()
    agree16 = torch.argmax(ref, -1).eq(torch.argmax(got16, -1)).float().mean().item()
    print(f"[bf16] max|logit diff| = {max16:.4e}   argmax agreement = {agree16:.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
