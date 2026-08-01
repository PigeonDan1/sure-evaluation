# scoring/vad_auc_roc

Computes ROC AUC from explicit VAD `frame_scores`.

The node samples frame centers at `frame_shift_sec` (default `0.01` seconds).
Each participating frame gets its label from reference `speech_segments` and
its score from the prediction `frame_scores` interval covering that center.
Frame centers without a score are excluded. If all participating frames have
only one label class, `auc_roc` is reported as `null` and the skip reason is
recorded. The node never falls back to hard labels from predicted
`speech_segments`.
