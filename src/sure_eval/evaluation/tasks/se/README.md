# SE Task

The SE task evaluates speech enhancement outputs.

Input rows use `enhanced_audio` for the system output, `noisy_audio` for the
model input when available, and `reference_audio` for clean speech when
full-reference metrics are selected.

Supported metrics:

| Family | Metric | Node |
| --- | --- | --- |
| Full-reference quality | `si-sdr` | `scoring/si_sdr` |
| Full-reference quality | `stoi` | `scoring/stoi` |
| Full-reference quality | `pesq` | `scoring/pesq` |
| No-reference quality | `dnsmos` | `scoring/dnsmos` |
| No-reference quality | `wv-mos` | `scoring/wv_mos` |
| No-reference quality | `utmos` | `scoring/utmos` |

All metrics are higher-is-better and aggregate by arithmetic mean over samples.
`pesq` and `stoi` require optional Python packages at runtime.
