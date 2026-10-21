# AudioLDM Eval

Computes corpus FAD, PANNs FD, paired softmax KL, and Inception Score for TTA.
Lower is better for FAD, FD, and KL; higher is better for IS. Generated and
reference rows are aligned by sample_id.

`sure-eval env setup --node scoring/audioldm_eval` automatically downloads
the pinned PANNs CNN14 checkpoints from Zenodo plus the pinned TorchVGGish
source archive, model weights, and PCA parameters used by FAD. The 16 kHz
evaluator uses `Cnn14_16k_mAP=0.438.pth`; the companion 32 kHz checkpoint is
also installed because upstream AudioLDM Eval checks for it during
initialization. Runtime scoring uses only these local assets and does not
contact GitHub. Set `AUDIOLDM_EVAL_CNN14_CHECKPOINT` to override the 16 kHz
checkpoint.
