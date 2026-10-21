# CLAP Score

Computes cosine similarity between LAION-CLAP embeddings of prediction_audio
and caption, followed by a corpus mean. Higher is better.

`sure-eval env setup --node scoring/clap_score` downloads only the pinned
`630k-audioset-best.pt` file from the official
[`lukewys/laion_clap`](https://huggingface.co/lukewys/laion_clap) repository
and verifies its SHA-256. Set `CLAP_MODEL_PATH` to use an existing compatible
checkpoint instead.
