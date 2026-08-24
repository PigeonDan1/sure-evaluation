# NeMo Arabic TN

This node applies NVIDIA NeMo's Arabic text normalization grammar to reference
and hypothesis ASR transcripts before CER scoring. Written tokens such as digits
are converted to spoken Arabic form. The node runs in a
node-local environment so the main SURE environment does not need `pynini` or
`nemo_text_processing` installed.

The selected profile is `ar_tn`, backed by the official
`nemo-text-processing==1.2.0` package. Run
`sure-eval env setup --node normalization/nemo_norm` to install it in the
node-local environment. The SURE wheel does not vendor NeMo source or grammar
data.

NeMo 1.2.0 declares `cdifflib` globally even though deterministic Arabic TN
does not use its audio-alignment helper. The node excludes that C extension
from `uv` resolution and provides the standard-library `difflib.SequenceMatcher`
API during NeMo's eager package import. This avoids a compiler and Python header
requirement without changing the Arabic TN graph or normalization path.
