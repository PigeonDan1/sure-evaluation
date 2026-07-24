# SE Task

The SE task evaluates speech enhancement outputs.

Input rows use `enhanced_audio` for the system output, `noisy_audio` for the
model input when available, and `reference_audio` for clean speech when
full-reference metrics are selected.

Supported metrics:

| Canonical metric | Execution selector | Node |
| --- | --- | --- |
| `si_sdr` | `si_sdr` / `si-sdr` | `scoring/si_sdr` |
| `stoi` | `stoi` | `scoring/stoi` |
| `pesq` | `pesq` | `scoring/pesq` |
| `dnsmos` | `dnsmos` | `scoring/dnsmos` |
| `wv_mos` | `wv_mos` / `wv-mos` | `scoring/wv_mos` |
| `utmos` | `utmos` | `scoring/utmos` |

All metrics are higher-is-better and aggregate by arithmetic mean over samples.
`pesq` and `stoi` require optional Python packages at runtime.
