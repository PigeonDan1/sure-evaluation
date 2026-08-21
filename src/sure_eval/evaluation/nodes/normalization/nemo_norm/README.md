# NeMo Arabic TN

This node applies NVIDIA NeMo's Arabic text normalization grammar to reference
and hypothesis ASR transcripts before CER scoring. Written tokens such as digits
are converted to spoken Arabic form. The node runs in a
node-local environment so the main SURE environment does not need `pynini` or
`nemo_text_processing` installed.

The selected profile is `ar_tn`, backed by `nemo_text_processing==1.2.0`.
