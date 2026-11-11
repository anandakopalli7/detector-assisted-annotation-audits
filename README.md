# Detector-assisted annotation audits

Reproducibility materials for **Detector-Assisted Annotation Audits: Estimation,
Interval Design, and Measurement Limits**, Ananda Siva Kumar Kopalli Naga Venkata,
Independent researcher. Journal acceptance is not claimed.

Archive identifier: https://doi.org/10.5281/zenodo.24126429
Code: https://github.com/anandakopalli7/programme-b-annotation-audits

## What is included

The revision combines the retained Paper 1 KITTI aggregate audit and Codex Paper
2 computational work with two fixed OJCS studies: 10,000 synthetic paths and
3,000 audit paths on three constructed COCO deletion populations. Claude Paper
2 is a separate manuscript and its bundle is not part of this repository.

The contribution separates point-estimation error, interval precision and
measurement validity. Observed precision improvements do not establish general
superiority. External point-estimation failures and the single-rater measurement
limitations are preserved. The 117 strong-method coverage tests and six paired
primary comparisons are fully reported, including inconclusive finite bounds.

## Files and installation

Use Python 3.12. In a virtual environment, install requirements.txt. This is
enough to analyze the supplied derived frames; no GPU or detector download is
needed. Rebuilding detections separately requires PyTorch 2.8.0 and TorchVision
0.23.0, the official dataset, and the recorded weights. No cloud setup is included.

Clone/download this code. Extract the inputs and results ZIPs from the archive
into the same root, preserving their 01_baseline, 03_protocol, 04_code and
05_results folders. Extract the checkpoint ZIP there only for full raw-data
checks. The four archives have a common relative layout and do not overwrite
one another. Dataset images, model weights, private administration and protected
human records are not supplied.

## Verification without new experiments

```text
python -B verify_artifacts.py --code-only
python -B verify_artifacts.py
python -B 01_baseline/codex_paper2/code_package/reproduce.py verify
python -B 01_baseline/paper1_source/anc/30_CODE/check_public_aggregates_v01.py 01_baseline/paper1_source/anc/50_RESULTS/PAPER1_COMPONENTS_v02.json
```

The full verify command requires both the inputs and results archives. For the
separate 13,000-path arithmetic reduction, also unpack the checkpoint archive
and run `python -B 04_code/ojcs_independent_review.py` in a disposable copy.
That check writes a fresh timing receipt in ANALYSIS_V1, changing its manifest
hash; it does not run scientific experiments. Original receipts stay in the
released archive. The agreement is a separate reduction of shared data, not
independent measurements or a second implementation of all inference methods.

ANALYSIS_V2 is authoritative. V1 is retained because a post-result proof check
corrected the two-sided tail allocation from log(240) to log(480). No raw data,
effect estimate, coverage result or conclusion changed. The erratum and both
analyses disclose this. The original analysis script produces historical V1;
do not interpret its old two-sided bounds as the final result.

## Optional bounded replay

The original scientific kernels, generators, seeds and count frames are retained
byte for byte. `reproduce_path.py` is a release-only local entry point, separating
scientific replay from private cloud/budget orchestration. A plan-only call is:

```text
python -B reproduce_path.py --cell M01 --replicate 0
```

Adding `--execute --max-seconds 600` explicitly runs that one fixed path locally.
It preserves originals, refuses existing replay outputs, uses no cloud API,
and stops its child at the deadline. Replay context hashes and machine timings
differ from the original run. All planned cell/replicate IDs are in the frozen
schedule; no new seed selection is needed. Repeating the study is unnecessary
for checking the supplied results. The historical runner and reducer are
preserved for provenance, not recommended as cloud-deployment commands.

## Licensing and evidence boundaries

See LICENSE and LICENSES_AND_DATA_PROVENANCE.md. Code and data-source rights are
distinct. COCO means provider-record deletion here, not independently verified
natural omissions. KITTI aggregate arithmetic does not validate human judgments.
The source manifest records administrative omissions explicitly; it does not
present a redacted artifact as byte-identical to the original execution bundle.
