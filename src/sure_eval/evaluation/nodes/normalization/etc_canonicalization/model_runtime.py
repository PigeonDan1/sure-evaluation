"""Minimal ETC-Qwen inference implementation kept inside the optional runtime."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are an ASR post-processing normalizer. Rewrite the given transcript into one "
    "deterministic shared written form so that reference and hypothesis transcripts of "
    "the same speech become identical: restore punctuation and casing, drop fillers, "
    "stutters and repetitions, and write numbers and symbols by written convention. "
    "Keep all other content unchanged. Output only the normalized text, with no "
    "explanation, quotes, JSON or markup."
)


def canonicalize_batch(texts: list[str], *, model: str) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    template = Path(model) / "chat_template.jinja"
    if template.exists():
        tokenizer.chat_template = template.read_text(encoding="utf-8")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    loaded = AutoModelForCausalLM.from_pretrained(
        model, dtype=dtype, local_files_only=True
    ).to(device).eval()
    loaded.config.use_cache = True
    outputs = []
    for text in texts:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = loaded.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        outputs.append(
            tokenizer.decode(
                generated[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
        )
    return outputs
