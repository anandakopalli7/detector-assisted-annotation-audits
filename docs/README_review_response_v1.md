# Detector-assisted annotation audits

Current manuscript: **Detector-Assisted Annotation Audits: Capture Gaps, Count
Estimation, and Review Allocation**, Ananda Siva Kumar Kopalli Naga Venkata,
Independent researcher. This is an author-review research draft; journal submission
and acceptance are not claimed.

## Current revision: review-response-v1

[Article](revisions/review-response-v1/ProgrammeB_OJCS_article_revised.pdf) |
[Supplement](revisions/review-response-v1/ProgrammeB_OJCS_supplement_revised.pdf) |
[TeX source](revisions/review-response-v1/ProgrammeB_OJCS_current_sources.zip)

The [current release](https://github.com/anandakopalli7/programme-b-annotation-audits/releases/tag/ojcs-review-response-v1)
contains the complete Supplementary Evidence ZIP. This repository is private.
Journal reviewers should receive that ZIP with the submission; a private link is
not a substitute for reviewer access. Public archive release remains pending.
The earlier reserved Zenodo identifier does **not** identify this revision's evidence.

The current work adds a count-matched natural-reference study, fixed-procedure
sensitivities, aligned detector-capture/risk diagnostics, and an exact public-input
allocation comparison. Large gains over one comparator do not imply universal
superiority: the allocation comparison changes the practical recommendation.
The original failed directional hypothesis and unfavorable results are retained.
Claude Paper 2 is separate and is not the current manuscript.

## Verify existing evidence without new experiments

Download and extract ProgrammeB_OJCS_current_evidence.zip into a new directory.
Use Python 3.12 and NumPy 2.5.4 for the recorded replay. From the extraction root:

```text
python -B 14_review_response_v01/verify_current_evidence.py
python -B 14_review_response_v01/verify_current_evidence.py --bootstrap
```

The checks verify the manifest, 1,200 saved raw blocks, 96 method MSEs, 24 exact
allocation risks, 16 aligned residual-risk comparisons and 84 bootstrap-precision
records. The second command also replays the critical uncapped allocation contrast.
No network call, image download, detector training or paid compute is required.
See [reproduction instructions](revisions/review-response-v1/REPRODUCE_CURRENT.txt).

The full archive SHA-256 is `d1a577aac0b0c4aa66c7ff30cd3f4a4e225128aa00159c6d34e9abe516dca71e` (107754287 bytes).
The [isolated verification receipt](revisions/review-response-v1/ISOLATED_VERIFICATION.json)
records what was actually checked. These are computational integrity checks, not
independent scientific replication or journal approval.

## History and licenses

Existing root-level code belongs to the original interval/measurement release.
Its [historical README](docs/README_original_release_v1.md) is preserved for provenance;
use the current release entry point for the revised paper. Original data, failed
gates and prior manuscript versions remain in the evidence archive.

Original code is MIT. Original documentation/results use the approved CC BY 4.0
terms. Adapted reference data retain the source CC BY-SA 4.0 terms; other third-party
terms and attribution remain applicable. Dataset images, weights and credentials
are excluded. The original-code license does not relicense third-party material.
