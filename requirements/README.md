# requirements — pinned environments

Three environments reproduce everything in this repo. Two are frozen
here; the third is locked in `infra/`.

| environment | pin file / lock | what runs on it | rebuild |
|---|---|---|---|
| sprints-1-2 pipeline venv (`../wildlife-monitor/.venv`) | `pipeline-venv.txt` + the OpenCV 5 source build (header notes) | everything: ETL artifacts, MOG2/CLAHE gate, cv2.dnn MDv6 detection, classifier v0, E2E union, G3/S5 evals, agent runners, dashboard | `uv venv && uv sync` in the sprints-1-2 repo, then its `scripts/build-opencv5.sh`, then `pip install -r requirements/pipeline-venv.txt` equivalents (see file headers) |
| legacy baseline venv (`./baseline-venv/`, gitignored) | `baseline-venv.txt` | ONLY the PyTorch + Pytorch-Wildlife lane of the S5 latency table (`eval/run_latency.py` → `eval/_pytorch_bench.py`) | `python -m venv baseline-venv` then install per the file header (CPU torch index) and the freeze |
| infra (`infra/`) | `infra/uv.lock` (uv, exact) + pins inside `infra/docker/Dockerfile.lambda-py312-opencv5` | ingest/motion Lambda unit tests (boto3 + moto), LocalStack smoke, the Lambda runtime image | `cd infra && uv sync` (tests: `uv run pytest -q`) |

Notes:

- The pipeline venv's `cv2` is a from-source OpenCV 5.0.0 build (tag
  5.0.0, minimal module set, bundled codecs) — there is no pip
  distribution to pin, so the build recipe reference lives in the
  `pipeline-venv.txt` header. `wildlife-monitor==0.1.0` in that freeze
  is the sprints-1-2 repo's own package, installed editable from
  `../wildlife-monitor`.
- The Lambda image pins its runtime deps explicitly
  (`awslambdaric==4.0.4`, `boto3==1.43.95`, `numpy==2.5.3`) and vendors
  the same source-built OpenCV 5 shared libs the pipeline venv uses —
  see the Dockerfile and `infra/scripts/build-lambda-image.sh`.
- Freeze date for both files: 2026-09-16 (the S5 measurement
  environment; numbers in `eval/S5-REPORT.md` were produced on exactly
  these versions).
