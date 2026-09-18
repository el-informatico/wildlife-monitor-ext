#!/usr/bin/env python3
"""Generate WildGuard_Vision_AWS_Compute_Grant_Proposal_v4.docx.

v4 = v3 (WildGuard_Vision_AWS_Compute_Grant_Proposal_v3.docx) structure and
wording, merged with four additions from the session's v2:
  1. title-block contact line (youtube.com/@jzavalaq)
  2. false-positive-load row in the evaluation table
  3. ASCII architecture diagram replaced by v2's prose data-flow paragraph
     (its final Terraform sentence is dropped: v3's own closing paragraph
      already states that wording)
  4. team-name-options footnote at the end

Run:  uv run --with python-docx python scripts/make_grant_proposal_v4.py
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

OUT = os.path.expanduser(
    "~/projects/wildlife-monitor-ext-deliverables/WildGuard_Vision_AWS_Compute_Grant_Proposal_v4.docx"
)


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

# ---- Title block (v3 + v2 contact line) ----
para(doc, "WildGuard Vision", size=20, bold=True, color=NAVY, after=2)
p = para(doc, "Wildlife Agentic Monitor — autonomous triage for camera-trap fleets", size=11.5, italic=True, color=GRAY, after=8)
hr(p, color=ACCENT, size="14")
para(doc, "OpenCV AI Competition 2026 · AWS Compute Grant ($150) · Agentic Vision · Solo team", after=1)
para(doc, "Juan Zavala Quijano · Software Engineer · Lima, Peru · September 17, 2026", after=1)
para(doc, "Contact: youtube.com/@jzavalaq", after=8)

# ---- Executive summary (v3) ----
heading(doc, "Executive Summary")
para(doc, "Camera traps generate far more imagery than field teams can review. WildGuard Vision turns that backlog into a short, auditable approval queue: OpenCV 5 gates, detects, and classifies frames; an agent then uses those visual results to decide whether to re-analyze, escalate, request human approval, or archive each event.")
para(doc, "The system is already built and measured. We processed 1,504/1,504 images end-to-end through the AWS ingest lane; evaluated detection on 800 images (recall 0.79); measured that 42.5% of frames never require detector inference; and verified 195/200 agent decisions against a rules-only baseline. The full AWS architecture is Terraform-pinned and LocalStack-tested.")
para(doc, "The grant funds the remaining real-AWS validation: replicate the ingest benchmark, run the 26,177-frame corpus, benchmark Graviton, and expose a judge-facing endpoint. In other words, the request is for cloud execution and reproducible validation—not exploratory development.")

# ---- 1. Problem (v3) ----
heading(doc, "1. Problem & Real-World Impact")
para(doc, "Wildlife-monitoring programs rely on camera traps, but the resulting image volume creates a review bottleneck. Empty frames, vegetation, illumination changes, and detector artifacts can bury rare but important evidence such as an endangered species or a person entering a restricted area.")
para(doc, "WildGuard Vision addresses the bottleneck by making the review queue evidence-driven. Every detection event receives exactly one disposition—re-analyze, escalate, human approval, or archive—and the rationale is retained for audit. On the 200-event evaluation corpus, the agent removed all 5 blind high-urgency escalations caused by probable detector artifacts while dropping zero person-related evidence.")
para(doc, "Privacy is architectural: images containing people require human approval before use, and faces are blurred on the image-serving path. The system is designed to approve evidence, not accuse people.")

# ---- 2. OpenCV 5 (v3) ----
heading(doc, "2. OpenCV 5: The Perception Engine")
para(doc, "OpenCV 5 is the substantive image-analysis layer, not a wrapper around an AI chatbot. It is used in three measurable roles:")
bullet(doc, [("Motion gating: ", True), ("MOG2 background subtraction with CLAHE-enhanced night preprocessing discards empty frames before detector inference. Measured result: 42.5% of frames avoid inference.", False)])
bullet(doc, [("Detection: ", True), ("MegaDetector v6 (YOLOv10 ONNX) runs through OpenCV DNN and produces animal, person, and vehicle boxes at a deployment confidence threshold of 0.25.", False)])
bullet(doc, [("Classification: ", True), ("a four-species ONNX classifier also runs through OpenCV DNN and reaches 0.987 top-1 accuracy on held-out crops.", False)])
para(doc, "OpenCV execution depth is measured internally: on 800 identical inputs and weights, the deployed new graph engine measured $3.67 per million images versus $4.16 for the PyTorch-based legacy lane—a 13% lower measured compute cost in that benchmark.")

# ---- 3. Agentic Vision (v3) ----
heading(doc, "3. Agentic Vision: Perception Changes What Happens Next")
para(doc, "WildGuard Vision is designed around the competition's Agentic Vision qualification: visual output changes a subsequent plan, tool call, action, or human-approval request.")
para(doc, "Example trace: a night frame produces a marginal animal detection at confidence 0.34. Instead of accepting the first result, the agent selects a night re-analysis action. OpenCV re-runs detection with an enhanced preprocessing recipe, surfaces four previously hidden low-confidence boxes, and the final disposition follows the second-pass evidence.")
para(doc, "The loop is therefore: PERCEIVE → DECIDE → RE-ANALYZE / ESCALATE / APPROVE / ARCHIVE → OBSERVE AGAIN WHEN NEEDED. The agent is bounded to a fixed action set, and every event closes with one auditable disposition.")

# ---- 4. AWS Architecture (v3, diagram → v2 prose) ----
heading(doc, "4. AWS Architecture & Deployment")
para(doc, "The AWS lane is specified as infrastructure-as-code and validated end-to-end against LocalStack. The Lambda container image boots the real AWS runtime with source-built OpenCV 5 vendored inside.")
para(doc, "Data flows from cameras to S3; object-created events invoke the validation and motion Lambdas, which write normalized records to DynamoDB; the triage agent consumes those records and closes every event with one action; the read-only dashboard exposes the approval queue, species counts, camera status, and a failure gallery.")
table(doc,
      ["AWS service", "Role"],
      [
          ["S3 (versioned)", "Raw media, detection sidecars, and motion-job objects"],
          ["AWS Lambda", "Validation/normalization and containerized OpenCV 5 motion analysis"],
          ["ECR", "Container image containing source-built OpenCV 5"],
          ["SQS", "Dead-letter handling for malformed input; 14-day retention, zero retries by policy"],
          ["DynamoDB", "Normalized detection and motion records, keyed for per-camera queries"],
          ["CloudWatch", "Operational logs and benchmark observability"],
      ],
      widths=[1.6, 4.9])
para(doc, "A signed Terraform plan defines a 14-resource deployment graph. The remaining blocker is not architecture design: the development environment has no AWS identity with the service permissions needed to run the planned real-AWS workload.")

# ---- 5. Evaluation (v3 + v2 FP row) ----
heading(doc, "5. Evaluation: What Is Already Proven")
table(doc,
      ["Metric", "Measured result"],
      [
          ["Ingest integrity", "1,504/1,504 images through S3 → Lambda → DynamoDB; zero missing, zero dead-lettered"],
          ["Detection", "800 images; IoU ≥ 0.5; TP/FP/FN 812/676/210; recall 0.79; night recall 0.85 vs 0.78 day"],
          ["False-positive load", "3.10 FP/camera/day (Missouri) vs 1.54 (African Wildlife); fleet mean 1.95"],
          ["Inference savings", "42.5% of frames do not reach detector inference"],
          ["Person detection", "In-domain held-out stills: precision 0.96 / recall 0.88; COCO val2017 out-of-domain: 0.55 / 0.38"],
          ["Agent decisions", "195/200 agreement with rules-only baseline; 5 divergences are blind escalations the agent downgrades to human review"],
          ["Testing", "189 offline tests + 90 infrastructure tests = 279 automated tests"],
      ],
      widths=[1.9, 4.6])
para(doc, "Known limitations are explicit. Vehicle detection is not claimed as reliable in-domain (0/60 annotated stills). The day/night partition is synthetic and seed-pinned. A seven-case failure catalog ships with the repository. These limitations will be part of the public evaluation rather than hidden behind headline metrics.")

# ---- 6. Grant (v3) ----
heading(doc, "6. Why the $150 Grant Is Needed")
para(doc, "The grant converts a tested LocalStack deployment into a reproducible real-AWS run that judges can inspect. The requested budget is deliberately tied to concrete workloads:")
table(doc,
      ["Workload", "Scale", "Est. cost"],
      [
          ["Real-AWS ingest replication", "≈1,500 container invocations; S3 + DynamoDB", "$5"],
          ["Full-corpus evaluation", "26,177 frames / 48 cameras; ≈26,000 invocations; ≈13 GB S3", "$40"],
          ["Judge-facing endpoint", "Lambda/API Gateway; optional small Graviton instance", "$15"],
          ["Graviton latency benchmark", "≈24 c7g instance-hours", "$5"],
          ["Scaling, memory, observability", "Configuration repeats, logs, transfer, operational margin", "$85"],
          ["Total", "", "≈ $150"],
      ],
      widths=[2.4, 3.2, 0.9], bold_last=True)
para(doc, "Approximately $60 is tied directly to the measured workloads above; the remaining amount provides room for sizing, observability, and repeatability experiments. Any unused credit will support the permanent public demonstration endpoint after the competition.")

# ---- 7. Build plan & judge demo (v3, incl. judge-demonstration paragraph) ----
heading(doc, "7. Build Plan & Judge Demonstration")
bullet(doc, [("Sep 21–Oct 2: ", True), ("Attend the required AWS grant check-in and replicate the 1,504-image ingest benchmark on real AWS.", False)])
bullet(doc, [("Oct 3–12: ", True), ("Run the 26,177-frame corpus on AWS and publish the resulting evaluation measurements.", False)])
bullet(doc, [("Oct 13–23: ", True), ("Bring the judge-facing endpoint online, record the Graviton benchmark, and finalize reproducibility documentation.", False)])
bullet(doc, [("Oct 24: ", True), ("Complete the Devpost submission, ahead of the Oct 26 deadline.", False)])
para(doc, "Judge demonstration: submit a known camera-trap event, show the OpenCV 5 evidence, trigger the agent's next action, and display the resulting disposition and audit trail. A second scenario demonstrates mandatory human approval for person-related evidence. The repository includes seven byte-deterministic agent scenarios whose expected decisions are asserted at runtime.")

# ---- 8. Team (v3) ----
heading(doc, "8. Team")
para(doc, "WildGuard Vision is a solo project led by Juan Zavala Quijano, a software engineer in Lima, Peru. The pipeline, AWS infrastructure, evaluation harness, agent, and dashboard are one person's verifiable work. The current implementation has 279 automated tests, and the application distinguishes measured results from unmeasured areas.")
para(doc, "Public engineering work and demonstrations are documented at youtube.com/@jzavalaq. The team bio is intentionally evidence-based: execution capacity is demonstrated through the completed pipeline, reproducible evaluation, infrastructure-as-code, and test coverage rather than by unverified claims.")

# ---- 9. Track (v3) ----
heading(doc, "9. Competition Track")
para(doc, "Primary path: Agentic Vision. WildGuard Vision meets the published qualification pattern because OpenCV 5 perception results influence the agent's subsequent tool/action choice: marginal detections trigger re-analysis, second-pass evidence can change the final disposition, and person detections route to mandatory human approval.")
para(doc, "The project is not entering the COOL path as its primary competition track. The planned Graviton measurement is an engineering benchmark and cloud-cost/latency validation, not a claim of COOL-award eligibility.")

# ---- Closing (v3) + v2 footnote ----
heading(doc, "Closing")
para(doc, "WildGuard Vision has completed the engineering needed to demonstrate the core idea: OpenCV 5 produces evidence, that evidence changes an agent's next action, and the resulting workflow remains bounded and human-controlled. The AWS Compute Grant supplies the missing execution environment for real-cloud validation, full-corpus measurement, and judge access. We will publish the resulting benchmarks and keep the demonstration available to the community after the competition.")

para(doc, "", after=2)
para(doc, "Team name options: WildGuard Vision (used in this document) · TrapTriage · SilentWatch — final name to be confirmed by the applicant at submission.", size=8.5, italic=True, color=GRAY, after=0)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
doc.save(OUT)
print("saved:", OUT)
