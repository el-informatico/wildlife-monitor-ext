#!/usr/bin/env python3
"""Generate AWS-GRANT-PROPOSAL-v2.docx — professional standalone grant application.

Run:  uv run --with python-docx python scripts/make_grant_proposal_v2.py
Output lands in ~/projects/wildlife-monitor-ext-deliverables/ (outside the repo).
"""
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

NAVY = RGBColor(0x0D, 0x2B, 0x45)
GRAY = RGBColor(0x55, 0x5F, 0x6E)
ACCENT = "2B6CB0"

OUT = os.path.expanduser("~/projects/wildlife-monitor-ext-deliverables/AWS-GRANT-PROPOSAL-v2.docx")


def hr(paragraph, color="C6D3E0", size="8"):
    pPr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), color)
    borders.append(bottom)
    pPr.append(borders)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def para(doc, text="", size=10.5, bold=False, italic=False, color=None, after=6, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    if align is not None:
        p.alignment = align
    if text:
        r = p.add_run(text)
        r.font.size = Pt(size)
        r.bold = bold
        r.italic = italic
        if color is not None:
            r.font.color.rgb = color
    return p


def rich(doc, chunks, after=6):
    """chunks: list of (text, bold) tuples."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    for text, bold in chunks:
        r = p.add_run(text)
        r.font.size = Pt(10.5)
        r.bold = bold
    return p


def heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(12.5)
    r.font.color.rgb = NAVY
    hr(p, color=ACCENT, size="10")
    return p


def bullet(doc, chunks, after=3):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.15
    for text, bold in chunks:
        r = p.add_run(text)
        r.font.size = Pt(10.5)
        r.bold = bold
    return p


def table(doc, headers, rows, widths=None, bold_last=False):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "0D2B45")
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.rows[i + 1].cells[j]
            c.text = ""
            r = c.paragraphs[0].add_run(val)
            r.font.size = Pt(9.5)
            is_total = bold_last and i == len(rows) - 1
            r.bold = bool(is_total or (j == 0 and len(headers) > 2))
            if is_total:
                shade(c, "EEF3F8")
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Inches(w)
    return t


doc = Document()
sec = doc.sections[0]
sec.top_margin = Inches(0.7)
sec.bottom_margin = Inches(0.7)
sec.left_margin = Inches(0.8)
sec.right_margin = Inches(0.8)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)

footer_p = sec.footer.paragraphs[0]
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
fr = footer_p.add_run("WildGuard Vision · AWS Compute Grant Application · OpenCV AI Competition 2026")
fr.font.size = Pt(8)
fr.font.color.rgb = GRAY

# ---- Title block ----
p = para(doc, "WildGuard Vision", size=20, bold=True, color=NAVY, after=2)
p = para(doc, "Wildlife Agentic Monitor — autonomous triage for camera-trap fleets", size=11.5, italic=True, color=GRAY, after=8)
hr(p, color=ACCENT, size="14")

for label, value in [
    ("Program: ", "OpenCV AI Competition 2026, powered by AWS — AWS Compute Grant ($150)"),
    ("Track: ", "Agentic Vision"),
    ("Applicant: ", "Juan Zavala Quijano, software engineer (Lima, Peru) — solo team"),
    ("Contact: ", "youtube.com/@jzavalaq"),
    ("Date: ", "September 17, 2026"),
]:
    rich(doc, [(label, True), (value, False)], after=1)
para(doc, "", after=2)

# ---- Executive summary ----
heading(doc, "Executive Summary")
para(doc, "Camera traps overwhelm field teams: parks collect more frames than any expert can review, and false positives bury the few images that matter. WildGuard Vision triages every frame autonomously — an OpenCV 5 pipeline gates, detects, and classifies each image, and a tool-calling agent then decides per event what to re-analyze, escalate, or archive, so that humans only approve a short evidence queue. The system is built and measured: 1,504 of 1,504 images processed end-to-end through the AWS ingest lane, detection evaluated on 800 images (recall 0.79, with 42.5% of frames never needing inference at all), and the agent's decisions verified against a rules-only baseline over 200 events. The full AWS architecture is already Terraform-pinned and LocalStack-tested; what is missing is precisely what this grant provides — the compute credits to run it on real AWS. We will use the $150 to replicate our ingest benchmark in the cloud, run the full-corpus evaluation, and host a judge-facing endpoint for the competition window.")

# ---- 1. Problem ----
heading(doc, "1. Problem and Real-World Impact")
para(doc, "Ecological monitoring programs depend on camera traps, and camera traps produce far more imagery than any team can review. Backlogs of weeks are routine; rare events — an endangered species, a person in a restricted zone at night — sit unread next to thousands of empty frames. The review bottleneck has two costs: missed evidence when queues grow too long, and wasted expert hours when every frame must be opened by hand.")
para(doc, "WildGuard Vision turns that firehose into a short approval queue. Every image is gated, analyzed, and classified automatically; each detection event receives exactly one disposition — re-analyze, escalate, queue for human approval, or archive — with the rationale recorded for audit. The impact for conservation teams and park rangers is direct: reviewers see the events that matter, with correct urgency, instead of everything. On our 200-event evaluation corpus, the agent eliminated 100% of the blind high-urgency escalations a conventional rules-only system fires on probable detector artifacts (5 of 5), while dropping zero person-related evidence — the same evidence reaches humans, at the right priority.")
para(doc, "Privacy is a design constraint, not an afterthought. Any image containing people requires human approval before use, faces are blurred on the only image-serving path without exception, and the product is built to approve evidence — never to accuse anyone.")

# ---- 2. Solution ----
heading(doc, "2. Solution and OpenCV 5 Image Analysis")
para(doc, "The perception layer is OpenCV 5, built from source and used as the substantive analysis engine in three roles:")
bullet(doc, [("Motion gating — ", True), ("MOG2 background subtraction with CLAHE-enhanced night preprocessing discards empty frames before any model runs; 42.5% of frames never reach the detector (measured inference saving).", False)])
bullet(doc, [("Detection — ", True), ("MegaDetector v6 (YOLOv10 ONNX) executed through OpenCV's DNN module on the new graph engine, producing per-frame animal, person, and vehicle boxes at a deployment confidence of 0.25.", False)])
bullet(doc, [("Classification — ", True), ("a four-species classifier (ONNX, also on OpenCV's DNN module) achieves 0.987 top-1 accuracy on held-out crops.", False)])
para(doc, "Engine depth is measured, not asserted: an ablation inside OpenCV itself (new graph engine vs. classic) over 800 images shows the deployed lane at $3.67 per million images versus $4.16 for a PyTorch-based legacy lane — a 13% advantage on identical inputs and weights.", after=6)
para(doc, "What makes the system agentic is that image results change subsequent actions — the Agentic Vision track's own qualification bar. In our committed demonstration traces, a marginal first-pass detection (animal, confidence 0.34, night frame) triggers a night re-analysis action: the system re-runs the detector live with a night-enhanced preprocessing recipe, surfaces four previously hidden low-confidence boxes, and the final disposition (archive) follows that second-pass evidence rather than the first. Perception drives the plan; the plan re-queries perception; the action follows the result.")

# ---- 3. AWS architecture ----
heading(doc, "3. AWS Architecture and Services")
para(doc, "The cloud lane is fully specified in Terraform and validated end-to-end against LocalStack; every component below is provisioned as code, and the Lambda container image boots the real AWS runtime with OpenCV 5 vendored inside.")
table(doc,
      ["Stage", "AWS service", "Role"],
      [
          ["Ingest", "S3 (versioned)", "Raw media, detection sidecars (versioned contract schema), and motion-job objects"],
          ["Validation", "AWS Lambda", "Detection-batch validation and normalization; idempotent transactional writes; poison messages routed to a dead-letter queue"],
          ["Motion analysis", "AWS Lambda (container image via ECR)", "Python 3.12 image vendoring the source-built OpenCV 5; runs the same MOG2 recipe in-function"],
          ["Buffering", "SQS (dead-letter queue)", "14-day retention for malformed input; zero retries by policy"],
          ["Storage", "DynamoDB (single table, on-demand)", "Normalized detection and motion records, keyed for per-camera queries"],
          ["Delivery", "S3 event notifications", "Object-created events per prefix drive both Lambda paths"],
      ],
      widths=[1.1, 1.9, 3.5])
para(doc, "Data flows from cameras to S3; object-created events invoke the validation and motion Lambdas, which write normalized records to DynamoDB; the triage agent consumes those records and closes every event with one action; the read-only dashboard exposes the approval queue, species counts, camera status, and a failure gallery. A real-AWS deployment path is prepared down to a signed Terraform plan (14-resource graph) and a boot-verified container image — the only missing element is an identity with service permissions, which these credits resolve.")

# ---- 4. Evaluation ----
heading(doc, "4. Evaluation and Measured Results")
para(doc, "All results are measured offline on CPU-only hardware; every figure traces to a committed evaluation report in the repository.")
table(doc,
      ["Metric", "Measured result"],
      [
          ["Ingest integrity (G1 benchmark)", "1,504 / 1,504 images end-to-end through S3 → Lambda → DynamoDB; zero missing, zero dead-lettered"],
          ["Detection (800 images, IoU ≥ 0.5)", "TP/FP/FN 812 / 676 / 210; recall 0.79; night recall exceeds day (0.85 vs 0.78)"],
          ["False-positive load", "3.10 FP/camera/day (Missouri) vs 1.54 (African Wildlife); fleet mean 1.95"],
          ["Person precision / recall", "In-domain held-out (1,740 stills): P 0.96 / R 0.88; out-of-domain ceiling (COCO val2017): P 0.55 / R 0.38"],
          ["Agent vs rules-only baseline", "195/200 agreement; all 5 divergences are blind escalations the agent correctly downgrades to human review"],
          ["Test suites", "189 offline tests (pipeline, evaluation, agent, dashboard) + 90 infrastructure tests"],
      ],
      widths=[2.2, 4.3])
para(doc, "We report limits as carefully as results: vehicle detection is measured as unreliable in-domain (0/60 annotated stills) and is therefore not claimed as a capability; a seven-case failure catalog with annotated images ships with the repository; and the day/night clock partition is synthetic (seed-pinned) over measured counts. Judges can verify the agent demo directly: seven scenarios with byte-deterministic traces in which every expected decision is asserted at run time, alongside the read-only operations dashboard with verified face-blur behavior.")

# ---- 5. Grant use ----
heading(doc, "5. Why We Need the Grant and Planned Use of Credits")
para(doc, "The project is deploy-ready, not deployed: the machine used for development carries no AWS identity with service permissions. The $150 grant converts a fully tested LocalStack architecture into a running real-AWS deployment for the competition window. Planned use (estimates at us-east-1 on-demand list pricing):")
table(doc,
      ["Workload", "Scale", "Est. cost"],
      [
          ["Replicate the 1,504-image ingest benchmark on real AWS", "≈1,500 container invocations (2 GB, ~45 s) + S3 storage + DynamoDB writes", "$5"],
          ["Full-corpus evaluation pass (26,177 frames, 48 cameras)", "≈26,000 invocations; ~13 GB in S3", "$40"],
          ["Judge-facing endpoint for the submission window", "On-demand Lambda/API Gateway; optional small Graviton instance for live sessions", "$15"],
          ["Graviton (c7g) latency benchmark", "≈24 instance-hours, completing the one cost cell we currently mark not-measured", "$5"],
          ["Scaling and memory-sizing experiments, observability logs, retries, transfer", "Repeat benchmarks across configurations; CloudWatch", "$85"],
          ["Total", "", "≈ $150"],
      ],
      widths=[2.6, 3.0, 0.9], bold_last=True)
para(doc, "Measured workloads account for roughly $60; the remainder is deliberate headroom for configuration experiments and operational safety margin. Any unspent credit still serves the project after the competition: the deployment becomes the permanent demonstration endpoint for the open-sourced system.")

# ---- 6. Milestones ----
heading(doc, "6. Plan and Milestones")
table(doc,
      ["Window", "Milestone"],
      [
          ["Sep 21 – Oct 2", "AWS grant check-ins attended (requirement for second credit tranche); real-AWS replication of the ingest benchmark"],
          ["Oct 3 – 12", "Full-corpus evaluation pass on real AWS; results folded into the public evaluation reports"],
          ["Oct 13 – 23", "Judge-facing endpoint live; Graviton benchmark recorded; final documentation"],
          ["Oct 24", "Devpost submission complete — 48 hours ahead of the October 26 hard deadline"],
      ],
      widths=[1.5, 5.0])

# ---- 7. Team ----
heading(doc, "7. Team")
para(doc, "WildGuard Vision is a solo team led by Juan Zavala Quijano, a software engineer based in Lima, Peru. Juan is currently competing across four concurrent hackathon tracks — including Ring/Amazon and AssemblyAI finalist-track projects — and documents his engineering work publicly at youtube.com/@jzavalaq. The entire system described here (pipeline, AWS infrastructure, evaluation harness, agent, and dashboard, with 279 automated tests) is one person's verifiable output; every claim in this application traces to a committed report, and gaps are stated as unmeasured rather than approximated. Team strength here is demonstrated throughput with measured-evidence discipline.")

# ---- 8. Track ----
heading(doc, "8. Competition Track")
para(doc, "We are competing in the Agentic Vision track. The qualification bar — image results must influence a subsequent plan, tool call, action, or request for human approval — is precisely the behavior our committed traces demonstrate: perception output selects the next analysis, live re-analysis changes the final disposition, and person detections route to mandatory human approval. We are not competing in the COOL track.")

# ---- Closing ----
heading(doc, "Closing")
para(doc, "The engineering is done and the evidence is committed; the credits fund the last mile — running a proven, privacy-first wildlife triage system on real AWS for the judges to operate themselves. We will attend every required check-in, publish all outcomes and benchmarks, and keep the deployed endpoint available to the community after the competition. Thank you for considering WildGuard Vision.")

para(doc, "", after=2)
p = para(doc, "Team name options: WildGuard Vision (used in this document) · TrapTriage · SilentWatch — final name to be confirmed by the applicant at submission.", size=8.5, italic=True, color=GRAY, after=0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("saved:", OUT)
