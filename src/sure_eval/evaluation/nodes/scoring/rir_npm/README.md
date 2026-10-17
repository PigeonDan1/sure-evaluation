# RIR NPM

## Purpose

Reports normalized projection misalignment for predicted/reference RIR pairs.
NPM is scale invariant and lower is better.

## Contract and computation

Input is \`rir_samples_jsonl\` with \`prediction_rir\`, \`reference_rir\`, and
\`sample_id\`. Each pair is flattened across time and channels. The reference is
projected onto the prediction with one least-squares scale, then the residual
ratio is reported as \`20 * log10(||reference - alpha * prediction|| /
||reference||)\` in dB.

## Runtime and limitations

The node runs in-process with NumPy and the optional audio loader. It does not
perform peak, delay, polarity, or channel alignment.

Source: SURE RIR task implementation in \`tasks/rir/metrics.py\`.
