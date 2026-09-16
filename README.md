# Wildlife Monitor (OpenCV 5 + AWS)

Camera-trap wildlife monitoring pipeline for the OpenCV AI Competition 2026
(Agentic Vision track). OpenCV 5 (built from source) runs MOG2 motion
filtering, CLAHE night preprocessing, and MDv6 ONNX detection via `cv2.dnn`;
results normalize into AWS for downstream triage.

License: Apache-2.0 (see `LICENSE`).

## Layout

- `infra/` — sprint-3 infrastructure foundation: S3 → Lambda → DynamoDB
  ingestion path as Terraform, LocalStack docker-compose, unit tests and a
  LocalStack smoke test. See `infra/README.md`.
- `pipeline/` — E2E union stage: MDv6 detections → training-recipe crops →
  classifier v0 species → contract-valid records over the ETL sample, with
  measured sanity metrics. See `pipeline/README.md` and `STATUS.md`.
- `eval/` — G3 perception evaluation: metrics harness with split
  integrity, day/night split, sequence logic, and a 7-case failure
  catalog. See `eval/G3-REPORT.md`.
- `agent/` — S4 triage agent (Gate G4): Bedrock-style tool loop over
  perception output with traced conditional actions, a rules-only
  kill-switch baseline, and a live second-pass night re-analysis demo.
  See `agent/README.md` and `agent/G4-REPORT.md`.
- `src/opencv/` — OpenCV 5 source tree used for the local build (untracked;
  fetch upstream).
- `data/`, `downloads/`, `models/`, `processed/`, `manifests/`, `build/`,
  `baseline-venv/` — local datasets, models and artifacts (untracked).

## Commit rules

Triple pre-commit validation is wired via `core.hooksPath scripts/hooks`
(enforced from the first commit):

1. No AI-tool attribution in commit messages (`scripts/hooks/commit-msg`).
2. No host paths, Windows-mount paths, or sibling-project identifiers in
   staged added lines (`scripts/guard-sensitive-content.sh`, token list
   lives untracked at `.git/sensitive-tokens`).
3. Never `git commit --no-verify` — the hooks are a tripwire, not
   containment.

No push or publication without explicit owner approval.
