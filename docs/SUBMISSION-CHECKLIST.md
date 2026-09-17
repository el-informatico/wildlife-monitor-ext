# Submission checklist — Devpost, hard deadline 26-Oct-2026 (Gate G6)

One submission, two repos published side by side (this repo + the
sprints-1-2 repo at `../wildlife-monitor` — ETL, detector port and
classifier training live there; see the conditional finding in §4).
Nothing has been pushed: every repo publication, video upload, and the
Devpost submission itself is local until the owner says YES (§3).
Gate text, verbatim: *"G6: submission completa en Devpost con ≥48h de
margen (submit ≤24-oct)"* — the Devpost entry must be COMPLETE by
24-Oct-2026. Companion docs in this packaging series:
`docs/TECH-REPORT.md`, `docs/ARCHITECTURE.md`, `docs/VIDEO-SCRIPT.md`.

## 1. Deadline math

Written 2026-09-16: 38 days to the submit-by date, 40 to the hard
deadline. All dates 2026; deadline clock is PT.

| date | milestone | owner | note |
|---|---|---|---|
| 09-16 (today) | this checklist; pre-publication audit run green (§4) | repo | audit re-verified same day, results in §4 |
| 10-20 | video recorded per `docs/VIDEO-SCRIPT.md` | human | also grab screenshots for §5 (d) in the same session |
| 10-22 | both repos publication-ready — audit re-run green on the exact tree to publish | repo + human | gated on §5 (a) publish decision |
| 10-23 | Devpost form drafted, complete, unsubmitted | human | field list in §3 (d) |
| 10-24 | **SUBMIT** — G6 met (≥48h margin) | human | nothing left but the button |
| 10-25 → 10-26 | buffer — only touched if Devpost or playback verification surfaces a problem | human | no planned work |
| 10-26 23:59 PT | **hard deadline** | — | competition |
| 10-27 → 11-09 | judging | — | competition |
| 10-27 → 11-09 | public voting | — | no prize; do not spend buffer here |

## 2. The 9 pieces — where each lives

Piece list verbatim from the brief (ES) with an English gloss. Status
vocabulary: READY (in-repo, done) · READY-local (done, local-only —
stated honestly, brief-compliant) · READY-BUT-HUMAN-GATED (in-repo part
done; a required step is human-only) · HUMAN-ONLY (cannot be done from
this repo at all).

| # | piece (verbatim ES) | EN gloss | artifact in this repo | status | owner |
|---|---|---|---|---|---|
| 1 | tech report | technical report | `docs/TECH-REPORT.md` | READY | repo |
| 2 | repo público o accesible al juez | public repo, or one accessible to judges | both repos side by side (this + `../wildlife-monitor`); publication gated on the owner YES | HUMAN-ONLY | human |
| 3 | deps pineadas + build/deploy/test | pinned dependencies + build/deploy/test | `requirements/` (two freezes, 2026-09-16) + `infra/uv.lock` + pins in `infra/docker/Dockerfile.lambda-py312-opencv5` + `docs/TECH-REPORT.md` §3 | READY | repo |
| 4 | diagrama de arquitectura | architecture diagram | `docs/ARCHITECTURE.md` | READY | repo |
| 5 | endpoint web funcional (o demo live agendada) | working web endpoint (or a scheduled live demo) | `dashboard/` — local read-only ops endpoint; run instructions in `dashboard/README.md`; no cloud deploy exists (eval/S5-REPORT.md §8) — the brief's scheduled-live-demo alternative is the fallback if judges need a live URL | READY-local | repo |
| 6 | video ≤5 min | video, 5 minutes max | script `docs/VIDEO-SCRIPT.md`; recording + upload are human | READY-BUT-HUMAN-GATED | script repo / record+upload human |
| 7 | evidencia de evaluación con casos de fallo y límites | evaluation evidence incl. failure cases and limits | `eval/G3-REPORT.md` + `eval/S5-REPORT.md` + `eval/failures/CATALOG.md` (7 cases, images committed under `eval/failures/images/`) | READY | repo |
| 8 | extras COOL | COOL extras | `eval/S5-REPORT.md` §5 — 3-lane latency/cost table (new vs legacy, +13% legacy overhead, $/1M with stated assumptions) + honest Graviton NOT MEASURED + documented bench procedure | READY (real Graviton bench optional → §5 b) | repo |
| 9 | extras Agentic: diagrama de flujo percepción→decisión→acción + TRACES donde la salida de OpenCV 5 cambia una decisión/tool-call/acción posterior + evaluación de éxito/fallo/observabilidad/control humano | Agentic extras: perception→decision→action flow diagram + traces where OpenCV 5 output changes a later decision/tool-call/action + success/failure/observability/human-control evaluation | flow in `docs/ARCHITECTURE.md`; committed traces `agent/demo/artifacts/` (`TRANSCRIPT.md` + `traces/` s1–s7 incl. `s2_pass2.json`: pass-1 → `night_reanalysis` → live pass-2 → archive) + `agent/artifacts/corpus_comparison.md` (kill-switch B) + `agent/G4-REPORT.md` | READY | repo |

All four `docs/` files land in the same packaging commit series — the
paths above are stable.

## 3. Human-gated actions (owner: human, in order)

1. Publication YES for both repos — and the one-repo vs two-repo
   decision. Default is two-repo side by side (one submission; audit
   note in §4). If one-repo is chosen, blocker §5 (a) fires FIRST:
   tokenize the clone-relative references and make the runtime import
   optional before anything is published.
2. Record the video per `docs/VIDEO-SCRIPT.md` (by 10-20).
3. Upload unlisted to YouTube; verify playback end-to-end, runtime
   ≤5:00.
4. Create/complete the Devpost entry: title · one-line pitch ·
   description (may embed/adapt `docs/TECH-REPORT.md`) · video URL ·
   BOTH repo URLs · team info · correct track selection (Agentic
   Vision) · any required checklists.
5. Final re-run of the §4 audit commands on the exact tree being
   published.
6. Press SUBMIT — by 10-24 (G6).
7. After submit: verify the rendered public page and the video
   playback.

## 4. Pre-publication audit

Four checks, verbatim commands, re-run 2026-09-16 against the frozen
tree for this checklist — results match the original audit run of the
same day. Nothing below is extrapolated.

| # | check | command (verbatim) | 2026-09-16 result |
|---|---|---|---|
| A | absolute/host paths | `git grep -nE '/home/\|/mnt/\|C:\\Users' -- .` | **0 hits** |
| B | AI-tool attribution — commit messages | `git log --format='%s%n%b' \| grep -icE 'co-authored-by\|claude\|anthropic\|copilot\|🤖'` | **0** |
| B′ | AI-tool attribution — content | `git grep -niE 'co-authored-by\|copilot' -- .` then broad `git grep -niE 'claude\|anthropic' -- .` | **0 attribution findings.** The token greps hit only `scripts/hooks/commit-msg` (the guard's own vendor-token regex — functional). The broad grep hits only `agent/adapters.py`: the string `anthropic.claude-sonnet-5-0` (a functional Bedrock model identifier — the documented adapter stub's default, kept deliberately) and one cost-note docstring phrase describing the model class to be measured. Neither is authorship attribution. |
| C | secrets | `git grep -nE 'aws_secret_access_key\|aws_access_key_id' -- .` | **0 real.** Only LocalStack dummy credentials (`aws_secret_access_key="test"`) in `infra/README.md` and `infra/scripts/localstack_smoke.py`, documented as test constants. `.env` ignored (`.gitignore`: `.env`, `infra/.env`); `infra/.env.example` tracked with placeholders; no cloud credentials ever in the repo. |
| D | sibling-project identifiers | `tr -d '\r' < "$(git rev-parse --absolute-git-dir)/sensitive-tokens" \| sed '/^[[:space:]]*$/d' \| git grep -IcFf - -- .` | **0 hits** against the full token list (list lives untracked inside the git dir — re-seed with a plain copy after a fresh clone) |

Environment sanity: `ls requirements/` → `README.md`,
`baseline-venv.txt`, `pipeline-venv.txt` (both freezes dated 2026-09-16
— the S5 measurement environment).

Conditional finding (owner: human decision): 13 tracked files reference
the sprints-1-2 repo in clone-relative form `../wildlife-monitor` (docs
and verbatim run commands), and `pipeline/e2e.py` resolves that path at
runtime (`import_wildlife`) to import the `wildlife_monitor` package.
Default resolution: publish BOTH repos side by side — they are one
submission. If the human decides to publish ONLY this repo, those
references must first be tokenized and the runtime import made
optional — tracked as blocker §5 (a).

The repo's own pre-commit guard already blocks these token classes on
every commit: `scripts/guard-sensitive-content.sh` (host paths +
sibling identifiers against the token list) and `scripts/hooks/`
(`pre-commit`, `commit-msg`), wired via `core.hooksPath` since the
first commit. Never `git commit --no-verify` — the hooks are a
tripwire, not containment.

## 5. Blockers / open items

| # | item | owner | due | blocks G6? | note |
|---|---|---|---|---|---|
| a | publish decision: two-repo (default) vs one-repo | human | before 10-22 | yes — gates publication-readiness | one-repo fires the §4 conditional finding: tokenize `../wildlife-monitor` references + make the `pipeline/e2e.py` runtime import optional, then re-run audit D |
| b | optional: real Graviton bench if AWS access resumes | human (AWS) | ideally before 10-24; skippable | no — optional per the brief | procedure in `eval/S5-REPORT.md` §5 (`--fresh` on c7g, arm64 wheels, same first-100 images, warmup 3 excluded); would replace the at-parity $2.99/1M arithmetic in `processed/s5_latency.md` |
| c | optional: Wellington person-labelled held-out ingest | human (data) | skippable | no | would upgrade person P/R from NOT_COMPUTABLE (`eval/S5-REPORT.md` §4); today's person numbers are load, not errors |
| d | Devpost form may require images/screenshots | human | take 10-20 during video recording; use at form drafting 10-23 | yes — form completeness | capture from the dashboard (map, species, approval queue, `X-Face-Blur` behavior) and the `eval/failures/` gallery in the same session as the video |

## 6. Submission-day runbook (ordered, terse)

0. Test suites green on the exact tree, before tagging (counts of
   record: 189 + 90):

       ../wildlife-monitor/.venv/bin/python -m pytest pipeline/tests eval/tests agent/tests dashboard/tests -q   # 189 passed
       cd infra && uv run pytest -q                                                          # 90 passed

1. Re-run the §4 audit on the exact tree being published → all four
   checks green.
2. Confirm tree == published tag: clean status, tag on HEAD, remote
   refs match what Devpost will link.
3. Devpost form complete — §3 (4) field list, nothing blank.
4. SUBMIT — by 10-24 (G6: ≥48h margin).
5. Verify the rendered public entry + video playback end-to-end.
6. Screenshot the receipt and the submitted page.
