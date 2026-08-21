# Arabic NeMo TN + CER Design

## Goal

Correct the Arabic ASR evaluation route so it uses NVIDIA NeMo Arabic text
normalization and character error rate scoring.

## Root Cause

The existing node imports `InverseNormalizer`, labels its profile `ar_itn`, and
routes Arabic evaluation to `scoring/wenet_wer`. That implements spoken-to-written
conversion followed by word-level scoring. The requested NeMo grammar lives under
`text_normalization/ar`: its cardinal graph maps written digits to spoken Arabic,
and its public `Normalizer` API returns spoken form.

## Data Flow

Both reference and hypothesis remain `<key>\t<text>` files. Each text field is
passed through `Normalizer(input_case="cased", lang="ar")`. The normalized files
then pass to `scoring/wenet_cer`, which removes whitespace and computes corpus-level
character edit distance. The reported score is `(sub + del + ins) / all`.

## Route Identity

The Arabic default metric becomes `cer`. The atomic route is
`asr.ar.cer.nemo_norm_ar_tn_v1.wenet_cer_v1`, composed of
`normalization/nemo_norm` profile `ar_tn` and `scoring/wenet_cer`.

## Source Layout

The node-local `ar/` directory must contain the code from NeMo's
`nemo_text_processing/text_normalization/ar`, not its inverse-normalization
counterpart. The vendored package remains available so shared NeMo modules and
data files resolve without adding dependencies to the root environment.

## Compatibility Boundary

This correction does not add custom Arabic orthographic normalization. Diacritics,
Arabic-specific punctuation, Alef variants, Ya/Alef Maqsura, and Ta Marbuta remain
subject to the existing NeMo TN output and WeNet CER tokenization behavior.

## Validation

Regression tests prove that `21` normalizes to `واحد وعشرون`, Arabic route
description resolves to TN + CER, and equivalent written/spoken inputs score zero
after TN. Existing ASR route, agent-plan, pipeline identity, lint, and wheel-build
checks must remain green.

## Delivery

One correction commit is pushed to both `fork/feat/arabic-nemo-itn` for GitHub PR
#8 and `origin/multilingual-version` for GitLab. Both remote branch SHAs must match
the local commit.
