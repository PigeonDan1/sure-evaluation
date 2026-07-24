# WavLM-Large SIM Scoring Node

`scoring/wavlm_large_sim` scores speaker similarity with a WavLM-large style
provider. It reports the canonical metric `spk_sim`; `sim/wavlm-large` is the
method selector recorded in `execution_metrics`.

The default provider follows the Seed-TTS-Eval WavLM-large speaker verification path:
`wavlm_large_finetune.pth` is loaded from this node's `checkpoints/` directory,
then WavLM hidden states are aggregated through the finetuned ECAPA-TDNN head and
scored by cosine similarity.

The node wraps SURE's shared SIM normalization and aggregation. Model loading stays inside the injected provider.

The finetuned checkpoint is a local runtime asset, not repository content.
Place it at `checkpoints/wavlm_large_finetune.pth` or set
`WAVLM_LARGE_SIM_CHECKPOINT`; checkpoint directories are ignored by git and
must not be committed.
