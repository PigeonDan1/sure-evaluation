# WeNet WER/CER/MER Scoring

This node wraps `sure_eval.evaluation.nodes.scoring.wenet_wer.wenet_compute_cer.compute_wer`.

The wrapped script is treated as a composite scoring backend because it includes
tokenization, optional character splitting, case normalization, tag stripping,
and edit-distance counting. The pipeline report records these internal stages
instead of pretending the backend is only a single edit-distance function.
