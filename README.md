# Detector-Assisted Annotation Audits

**Capture Gaps, Count Estimation, and Review Allocation**

Ananda Siva Kumar Kopalli Naga Venkata, Independent researcher, Troy, MI, USA.
ORCID: https://orcid.org/0009-0002-5241-9922

## Current package: submission-v1

[Article](revisions/submission-v1/ProgrammeB_OJCS_article.pdf) |
[Supplement](revisions/submission-v1/ProgrammeB_OJCS_supplement.pdf) |
[Manuscript source](revisions/submission-v1/ProgrammeB_OJCS_sources.zip) |
[Full evidence release](https://github.com/anandakopalli7/programme-b-annotation-audits/releases/tag/ojcs-submission-v1)

Artifact DOI allocated for this exact release: https://doi.org/10.5281/zenodo.24126429
Zenodo publication is pending user upload and verification at release preparation.
This is an author manuscript and research-artifact release. Journal submission
and acceptance are not claimed by this README.

The study separates detector capture, finite-budget count estimation and review
allocation using matched natural/synthetic controls. Established estimators and
source annotations are credited. The failed original directional hypothesis,
adverse comparisons and post hoc labels remain. Results concern operational
reference counts on fixed cohorts, not physical completeness or universal gains.

## Code, data and reproduction

Download **ProgrammeB_OJCS_evidence.zip** from the release and extract it into a
new directory. The ZIP includes current and historical study code, derived input
frames, retained detector predictions, raw result blocks, frozen protocols and
analysis. The current entry point is `START_HERE_CURRENT.txt`. From its root:

```text
python -B 18_submission_release_v01/verify_submission_package.py
```

Use Python 3.12 with NumPy. The saved exact bootstrap replay environment uses
NumPy 2.5.4. The read-only check validates 9,138 manifest members and rechecks
1,200 raw blocks, 96 method means, 24 exact allocation risks, 16 aligned residual
risk comparisons and 84 precision records. No cloud or image download is needed.
This is computational verification, not independent semantic reannotation.

Archive SHA-256: `d13547ed90f7fc7ad807c927044b692868130957f9aed52b7c884abb20daf24b`

Scientific text, equations, numeric tables and figure bytes are unchanged from
the corrected editorial-final-v1 paper. Only release availability metadata was
updated. See the verification receipts in `revisions/submission-v1/`.

## Licenses and history

Original code is MIT; original documentation/results are CC BY 4.0. Adapted
soft-label references retain CC BY-SA 4.0. Historical KITTI-related material
retains its recorded noncommercial/share-alike restrictions. See
[license and provenance mapping](revisions/submission-v1/LICENSES_AND_DATA_PROVENANCE.txt).
Images, weights and credentials are excluded.

Earlier root-level code and releases are retained for provenance; they are not
the entry point for the current paper. The previous README is preserved in
`docs/README_review_response_v1.md`. Historical author-side mock reviews are not
official or independent IEEE decisions. Claude Paper 2 is a separate project.
