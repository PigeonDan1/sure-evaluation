# NeMo Arabic ITN

This node applies NVIDIA NeMo's Arabic inverse text normalization grammar to
reference and hypothesis ASR transcripts before WER scoring. It runs in a
node-local environment so the main SURE environment does not need `pynini` or
`nemo_text_processing` installed.

The selected profile is `ar_itn`, backed by `nemo_text_processing==1.2.0`.
