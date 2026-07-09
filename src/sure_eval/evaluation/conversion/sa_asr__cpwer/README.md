# SA-ASR cpWER Conversion

This conversion profile links SA-ASR STM inputs to the key-text normalization
interface and then back to STM for MeetEval scoring.

It is not a metric node. The SA-ASR task route records it in
`conversion_trace` because input format conversion can affect metric results.
