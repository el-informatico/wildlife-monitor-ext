# DEPLOY-AWS.md — real-AWS deployment runbook (T-WL-1)

## State summary 2026-09-17 20:35

- **Diff (uncommitted at write time):** terraform gains a real-AWS target —
  `aws.tfvars` (`aws_endpoint=""` + account-qualified names + ECR image URI),
  a `motion_image_uri` switch that deploys the motion Lambda as an ECR
  container (`lambda.tf` zip/Image count-variants, paired invoke configs,
  account-scoped log policy via `data.aws_caller_identity`), outputs for the
  motion function; plus this runbook, `scripts/push_lambda_image.py` (boto3 —
  no aws CLI here), the AWS-variant Dockerfile, `terraform/iam-deploy-policy.json`
  (least-privilege to attach), `evidence/aws/*`, and pointers in README/TECH-REPORT.
- **Verified offline today:** docker build + boot check of `:g1-aws` (cv2 5.0.0,
  handler, RIC, MOG2 — `evidence/aws/image-bootcheck.txt`); `terraform validate`;
  real-AWS dry-run `plan -refresh=false` in workspace `aws` → **14 to add**,
  STS read `id=201023850731` (`evidence/aws/plan-aws.log`); LocalStack default
  workspace plan → **0 drift**; moto suite → **90 passed**.
- **NOT verified (needs a permitted identity):** ECR push, `terraform apply`,
  the real G1 pass, every cost number in §7 (estimates only).
- **Remains in-session:** STATUS.md entry, local commit, T-WL-5 verification.

**Status: NOT DEPLOYED to real AWS. Deploy-ready.** Everything below that
could be validated without service permissions *was* validated locally
(§2); the one blocker is the IAM identity on this machine (§1). This file
is the exact, ordered path from "permissions granted" to "measured real-AWS
G1 pass + evidence", with costs (§7) and teardown (§8).

The LocalStack deployment (G1's 1,504/1,504 measured pass,
`G1-AWS-REPORT.md`) remains the measured reference; this lane covers the
same terraform against real regional endpoints.

---

## 1. Credentials finding (2026-09-17, this machine)

`~/.aws/credentials` holds one profile, `[default]`, whose keys ARE valid:
every probe signed correctly and STS answered —
`arn:aws:iam::201023850731:user/hackathon-deploy`, region us-east-1.

That user has **zero service permissions**: no policies attached. Every
service call is denied — read-only probes (s3:ListBuckets,
dynamodb:ListTables, sqs:ListQueues, lambda:ListFunctions,
logs:DescribeLogGroups, ecr:DescribeRepositories, iam:ListAttachedUserPolicies,
iam:SimulatePrincipalPolicy) and a decisive write probe
(s3:CreateBucket on a throwaway name). Only `sts:GetCallerIdentity`
succeeds. (Secrets were never printed; only ARN/masked fragments.)

Consequence, honestly labeled: **no real-AWS deploy, invocation, or cost
measurement happened.** §7 costs are estimates from public us-east-1
pricing, not measurements.

## 2. What WAS validated locally (evidence on disk)

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Motion-Lambda container image builds and boots with the real AWS entrypoint (`_HANDLER` + `LAMBDA_TASK_ROOT` baked; `bootstrap.py` runs its own path setup; cv2 5.0.0, `motion_app` import, `awslambdaric`, MOG2 construct) | PASS — python 3.12.3, image 279 MB | `evidence/aws/image-bootcheck.txt` |
| 2 | `terraform validate` on the real-AWS variant | PASS (pre-existing deprecation warnings only) | — |
| 3 | **Real-AWS dry run**: `terraform plan -refresh=false -var-file=aws.tfvars` in workspace `aws` — signs with the real default credentials, resolves `data.aws_caller_identity` against real STS (`Read complete … id=201023850731`), computes the full 14-resource graph | PASS — `Plan: 14 to add, 0 to change, 0 to destroy` | `evidence/aws/plan-aws.log`, `plan-aws-summary.json` |
| 4 | LocalStack no-regression: plan (default workspace, deployed stack) after the real-AWS edits | PASS — `0 to add, 0 to change, 0 to destroy` | §this section, re-runnable |
| 5 | moto suite (`uv run pytest -q` in `infra/`) | PASS — 90 passed | — |

Raw plan artifacts (`.tfplan`/`.json`) also sit in git-ignored
`build/aws-evidence/`; the tracked copies above are the durable record.

## 3. Unblock: grant the deploy identity its permissions (once)

The policy file `terraform/iam-deploy-policy.json` is least-privilege for
exactly this stack: the named bucket/table/queue/functions/role, PassRole
restricted to `lambda.amazonaws.com`, ECR push, log-group read-only for
evidence capture, plus the delete actions teardown needs.

Attach it (any machine with an admin identity and the `aws` CLI — this one
has no `aws` binary):

```bash
aws iam put-user-policy \
  --user-name hackathon-deploy \
  --policy-name wfm-deploy-twl1 \
  --policy-document file://terraform/iam-deploy-policy.json
```

Console equivalent: IAM → Users → hackathon-deploy → Add permissions →
Attach policies directly → Create policy → JSON → paste the file → attach.
Detach with `aws iam delete-user-policy --user-name hackathon-deploy
--policy-name wfm-deploy-twl1` after the run.

## 4. Deploy sequence (ordered, ~15 min hands-off)

All commands from `infra/`. `../baseline-venv/bin/python` (repo root) is
the interpreter with boto3 on this machine (no `aws` CLI installed).

```bash
# 0) sanity: the identity now has permissions
../baseline-venv/bin/python -c "import boto3; print(boto3.client('s3').list_buckets()['Owner'])"

# 1) build + push the motion image to ECR (idempotent; creates the repo)
../baseline-venv/bin/python scripts/push_lambda_image.py
#   → prints PUSHED <acct>.dkr.ecr.us-east-1.amazonaws.com/wfm-lambda-py312-opencv5:g1 + digest
#   terraform/aws.tfvars already pins exactly this URI

# 2) terraform: separate workspace keeps the LocalStack state untouched
cd terraform
../.tools/terraform init -input=false
../.tools/terraform workspace select aws        # created 2026-09-17, still empty
../.tools/terraform plan  -var-file=aws.tfvars  # now with refresh against real AWS
../.tools/terraform apply -var-file=aws.tfvars
../.tools/terraform output                     # bucket/table/function ARNs → evidence
cd ..
```

Why container packaging for motion on real AWS (zip on LocalStack): the
motion gate needs the source-built OpenCV 5 (glibc 2.39), which cannot fit
a 250 MB Lambda layer zip; `motion_image_uri != ""` switches the same
module to `package_type = "Image"` (`lambda.tf`), and LocalStack keeps its
runtime-image-mapping zip path — one terraform root, two targets.

## 5. Re-run the G1 pass against real AWS (the measured run)

```bash
# smoke slice first (~5% of the manifest), then full
../baseline-venv/bin/python scripts/g1_batch_load.py \
  --manifest <path-from-export_manifest_json.py> \
  --endpoint ""  --bucket wfm-raw-media-201023850731  --table wfm-detections-aws \
  --function-name wfm-motion-gate-aws  --dlq-name wfm-detection-ingest-aws-dlq \
  --limit 100
# full run: drop --limit; report lands in build/g1-load-report.json (overwrite
# the LocalStack one only after copying it aside — see G1-AWS-REPORT.md)
```

`--endpoint ""` is the real-AWS switch (boto3 then uses normal regional
endpoints); the Lambda env var equivalent is already empty in aws.tfvars.

## 6. Evidence to capture during/after the run

- `terraform output -json` (ARNs, names) → `evidence/aws/outputs.json`
- Invocation + error counts: Lambda console → both functions → Monitoring
  (Invocations / Errors / Duration p50/p100 / cold-start share)
- CloudWatch tail (the policy above allows read-only on `/aws/lambda/wfm-*`):
  `logs:GetLogEvents` on the motion group around the run window — save
  first/last ~50 lines as `evidence/aws/motion-log-excerpt.txt`
- DynamoDB counts: `dynamodb:Scan`/console item count for `wfm-detections-aws`
  vs the LocalStack baseline in `G1-AWS-REPORT.md`
- Update `G1-AWS-REPORT.md` §verdict: replace "LocalStack only" with the
  measured real-AWS numbers, side by side.

## 7. Cost (ESTIMATES — public us-east-1 pricing, not measured)

One full G1 pass (~1,500 frames, ~30 motion jobs, ~100 ingest batches):

| Service | Driver | Estimate |
|---|---|---|
| Lambda motion | ~30 jobs × ~200 s × 1 GB (x86) ≈ 6,000 GB-s | ~$0.10 |
| Lambda ingest | ~100 × ~2 s × 256 MB + requests | <$0.01 |
| DynamoDB on-demand | ~50k item writes + reads | ~$0.05 |
| S3 | ~1 GB stored + PUTs | ~$0.03 |
| ECR | 0.28 GB image storage | <$0.01/mo |
| CloudWatch Logs | motion logs, est. 0.2–1 GB ingest | $0.10–0.50 |
| **One-shot total** | | **≈ $0.30–0.70** |

Idle with the stack left up: ≈ **$0.05–0.10/month** (storage only; no
traffic). After teardown (§8): $0, except ECR if the repo is kept.

## 8. Teardown

```bash
cd terraform
../.tools/terraform workspace select aws
../.tools/terraform destroy -var-file=aws.tfvars   # bucket must be emptied first:
# baseline-venv/bin/python - <<'EOF'   (or the loader's cleanup phase)
#   boto3.resource('s3').Bucket('wfm-raw-media-201023850731').objects.delete()
#   boto3.client('s3').delete_bucket(Bucket='wfm-raw-media-201023850731')
# EOF
../.tools/terraform workspace select default && ../.tools/terraform workspace delete aws
```

`force_destroy` is intentionally off on the bucket (README caveat), hence
the manual empty step. Keep the ECR repo if a re-run is likely — it is the
279 MB artifact everything else rebuilds from.
