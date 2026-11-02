"""Dependency-free generation budget checks shared by runtime and tests."""

MAX_NEW_TOKENS = 256


def ensure_generation_completed(generated_count: int, last_token: int, eos_token_id) -> None:
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    if generated_count >= MAX_NEW_TOKENS and last_token not in eos_ids:
        raise RuntimeError(
            f"ETC canonicalizer exhausted max_new_tokens={MAX_NEW_TOKENS} without EOS; "
            "input requires a shorter transcript or a separately configured runtime budget"
        )
