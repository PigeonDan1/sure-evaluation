# TTA

Text-to-audio evaluation with AudioLDM-Eval FAD/FD/KL/IS, LAION-CLAP text-audio similarity, and AudioBox Aesthetics CE/CU/PC/PQ.

Input uses samples JSONL. Every row requires sample_id and prediction_audio. FAD/FD/KL/IS also require reference_audio; CLAP also requires caption.

Lower is better for FAD, FD, and KL. Higher is better for IS, CLAP and all four AudioBox axes.
