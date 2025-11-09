import os
import io
import base64
import json
import datetime
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

from google import genai      # pip install google-genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import hashlib
import time
from pathlib import Path
from auth import auth_bp, login_required, init_db

import prompts  # Custom prompts module

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch 
from reportlab.platypus import Image as RLImage, PageBreak
from reportlab.pdfgen import canvas


env = load_dotenv()

# ------------------------ Runtime Tunables (env) ------------------------
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1024"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() in ("1", "true", "yes")
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache/detections"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
USE_ONE_LINER_MODEL = os.getenv("USE_ONE_LINER_MODEL", "false").lower() in ("1", "true", "yes")


# ------------------------
# Config
# ------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set. Set it before running or Gemini calls will fail.")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-only-key-CHANGE-THIS-IN-PRODUCTION")
CORS(app)

# Register auth blueprint and initialize database
app.register_blueprint(auth_bp, url_prefix='/auth')
init_db()

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

ALLOWED_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}


# ------------------------
# Helpers
# ------------------------
def allowed_file(fname: str) -> bool:
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def detect_from_bytes(raw: bytes) -> Dict[str, Any]:
    """Run the same detection logic as the /detect route but as a callable function.

    Returns a dict with keys: detections, annotated_image_b64, wrong, right, todo
    Raises RuntimeError on Gemini or parsing errors.
    """
    # Resize/compress image to reduce payload for low-network environments
    pil = pil_from_bytes(raw)
    w, h = pil.size
    # Only resize if larger than MAX_IMAGE_DIM
    max_dim = MAX_IMAGE_DIM
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    prompt_text = prompts.auditor(pil.size[0], pil.size[1])

    # Use a cache key based on image bytes + prompt text so repeated uploads avoid re-calling the model
    cache_key = hashlib.sha256()
    try:
        # Use compressed bytes for the key (consistent)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=JPEG_QUALITY)
        compressed = buf.getvalue()
        cache_key.update(compressed)
    except Exception:
        cache_key.update(raw)
    cache_key.update(prompt_text.encode("utf-8"))
    key_hex = cache_key.hexdigest()

    if CACHE_ENABLED:
        cache_file = CACHE_DIR / f"{key_hex}.json"
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                    return cached
            except Exception:
                # On any cache read error, fall through and regenerate
                pass

    # Gemini call with basic retry/backoff to survive transient network issues
    retries = 2
    backoff = 1.0
    answer_text = ""
    for attempt in range(retries + 1):
        try:
            # send the compressed PIL to Gemini to reduce payload
            g_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[pil, prompt_text]
            )
            answer_text = getattr(g_response, "text", "") or ""
            break
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Gemini request failed: {e}")
            time.sleep(backoff)
            backoff *= 2

    try:
        cleaned = answer_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1].replace("json", "", 1).strip()
        analysis = json.loads(cleaned)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from Gemini: {e}; raw={answer_text}")

    detections = []
    for w_item in analysis.get("wrong", []):
        box = w_item.get("box")
        if box and len(box) == 4:
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append({
                "label": w_item.get("heading", "Hazard"),
                "confidence": w_item.get("severity", 0),
                "box": [x1, y1, x2, y2]
            })

    annotated = draw_boxes(pil, detections)
    annotated_b64 = img_to_b64(annotated)

    out = {
        "detections": detections,
        "annotated_image_b64": annotated_b64,
        "wrong": analysis.get("wrong", []),
        "right": analysis.get("right", []),
        "todo": analysis.get("todo", [])
    }

    if CACHE_ENABLED:
        try:
            with (CACHE_DIR / f"{key_hex}.json").open("w", encoding="utf-8") as fh:
                json.dump(out, fh)
        except Exception:
            pass

    return out

def img_to_b64(pil_img: Image.Image) -> str:
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG", quality=90)
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except Exception:
        return draw.textsize(text, font=font)

def draw_boxes(pil_img: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['label']} {det['confidence']:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        tw, th = text_size(draw, label, font)
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill="red")
        draw.text((x1 + 4, y1 - th - 4), label, fill="white", font=font)

    return img

def add_header_footer(canvas_obj, doc, company_name, site_name, logo_stream=None):
    """Draw header, footer, and optional logo."""
    canvas_obj.saveState()
    width, height = A4

    # --- Header text ---
    header_text = f"{company_name} | Site: {site_name}"
    canvas_obj.setFont("Helvetica-Bold", 10)
    canvas_obj.drawString(40, height - 40, header_text)

    # --- Optional logo ---
    if logo_stream:
        try:
            logo_stream.seek(0)
            canvas_obj.drawImage(logo_stream, width - 120, height - 70, width=60, height=40, preserveAspectRatio=True, mask='auto')
        except Exception as e:
            print("Logo render failed:", e)

    # --- Footer with page number ---
    page_num = doc.page
    footer_text = f"Page {page_num}"
    canvas_obj.setFont("Helvetica-Oblique", 9)
    canvas_obj.drawRightString(width - 40, 30, footer_text)

    canvas_obj.restoreState()

# ------------------------
# Routes
# ------------------------
@app.route("/")
@login_required
def home():
    return send_file("template/index.html")


@app.route("/detect", methods=["POST"])
@login_required
def detect():
    if "image" not in request.files:
        return jsonify({"error": "no image file provided"}), 400

    f = request.files["image"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "invalid or unsupported file"}), 400
    raw = f.read()
    try:
        result = detect_from_bytes(raw)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(result)

#------------ Multi Image Report Generation ------------

@app.route("/multi_report", methods=["POST"])
@login_required
def multi_report():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    company_name = request.form.get("company_name", "Not Specified")
    site_name = request.form.get("site_name", "Not Specified")
    site_location = request.form.get("location", "Not specified")
    audit_date = request.form.get("audit_date", str(datetime.date.today()))
    logo_file = request.files.get("company_logo")  # Optional logo
    logo_stream = None

    if logo_file:
        try:
            logo_stream = io.BytesIO(logo_file.read())
        except Exception:
            logo_stream = None

    images = request.files.getlist("images")
    all_findings, summary_points, annotated_thumbnails = [], [], {}

    # Read raw bytes for all images first (safer for parallel processing)
    raw_images = []
    for idx, f in enumerate(images, start=1):
        try:
            data_bytes = f.read()
        except Exception:
            data_bytes = None
        raw_images.append((idx, data_bytes))

    # Parallelize detection to reduce wall-clock time (if network/model supports concurrency)
    max_workers = min(MAX_WORKERS, max(1, len(raw_images)))
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, data_bytes in raw_images:
            if not data_bytes:
                continue
            futures.append((idx, ex.submit(detect_from_bytes, data_bytes)))

        for idx, fut in futures:
            try:
                data = fut.result()
            except Exception as e:
                print(f"Detection failed for image {idx}: {e}")
                continue

            if data.get("annotated_image_b64"):
                try:
                    img_data = base64.b64decode(data["annotated_image_b64"])
                    annotated_thumbnails[idx] = io.BytesIO(img_data)
                except Exception:
                    pass

            for w_item in data.get("wrong", []):
                all_findings.append([
                    str(idx),
                    str(w_item.get("severity", "")),
                    idx,
                    w_item.get("heading", ""),
                    w_item.get("explanation", "")
                ])
                summary_points.append(w_item.get("heading", ""))

    # --- PDF Setup ---
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    # --- Header section ---
    story.append(Paragraph("<b>SAFETY AUDIT REPORT</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Company:</b> {company_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Site:</b> {site_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Location:</b> {site_location}", styles["Normal"]))
    story.append(Paragraph(f"<b>Audit Date:</b> {audit_date}", styles["Normal"]))
    story.append(Paragraph(f"<b>Report Generated:</b> {datetime.datetime.now()}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # --- Summary ---
    try:
        # prompts.multi_line_summary contains a {context} placeholder; replace it safely
        raw_prompt = prompts.multi_line_summary(summary_points)
        summary_prompt = raw_prompt.replace("{context}", json.dumps(summary_points))
        g_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[summary_prompt]
        )
        summary_text = getattr(g_response, "text", "") or ""
    except Exception:
        summary_text = "Summary generation failed."

    story.append(Paragraph(summary_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    # --- Table creation ---
    if all_findings:
        table_data = [["S.No", "Severity", "Image", "Key Findings", "Explanation"]]

        # Choose one-liner generation strategy: heuristic (fast, zero-network) or LLM batches
        one_liners = []
        if USE_ONE_LINER_MODEL:
            try:
                contexts = []
                for row in all_findings:
                    _, _, _, heading, explanation = row
                    contexts.append({"heading": heading, "explanation": explanation})

                batch_size = int(os.getenv("ONE_LINER_BATCH_SIZE", "20"))
                batch_one_liners = []
                for i in range(0, len(contexts), batch_size):
                    chunk = contexts[i:i+batch_size]
                    batch_prompt = (
                        "You are an expert content generator. For each item in the input list, produce a concise single-line summary (max 10 words)."
                        " Return a JSON array of strings in the same order as the inputs.\n\nINPUT: " + json.dumps(chunk)
                    )
                    try:
                        resp = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[batch_prompt]
                        )
                        txt = getattr(resp, "text", "") or ""
                        if txt.strip().startswith("```"):
                            txt = txt.split("```")[1]
                        parsed = json.loads(txt)
                        if isinstance(parsed, list):
                            batch_one_liners.extend(parsed)
                    except Exception:
                        for item in chunk:
                            h = item.get("heading", "")
                            e = item.get("explanation", "")
                            batch_one_liners.append((e.split(".")[0] + ".") if e else (h or ""))

                one_liners = batch_one_liners
            except Exception:
                one_liners = []
        else:
            # Heuristic: heading + first sentence of explanation (very fast, zero network)
            for _, _, _, heading, explanation in all_findings:
                if explanation:
                    one_liners.append((explanation.split(".")[0] + ".") if "." in explanation else explanation)
                else:
                    one_liners.append(heading or "")

        for idx, row in enumerate(all_findings, start=1):
            sno, severity, img_idx, heading, explanation = row

            # Prefer batch/heuristic-generated one-liners when available
            try:
                one_liner = one_liners[idx-1] if idx-1 < len(one_liners) and one_liners[idx-1] else (explanation.split(".")[0] + ".")
            except Exception:
                one_liner = explanation.split(".")[0] + "."

            if img_idx in annotated_thumbnails:
                annotated_thumbnails[img_idx].seek(0)
                thumb = RLImage(annotated_thumbnails[img_idx], width=0.9*inch, height=0.9*inch)
            else:
                thumb = Paragraph("—", styles["Normal"])

            table_data.append([
                sno,
                severity,
                thumb,
                Paragraph(heading, styles["Normal"]),
                Paragraph(one_liner, styles["Normal"])
            ])

            # --- Page break after 10 rows ---
            if idx % 10 == 0 and idx < len(all_findings):
                col_widths = [0.6*inch, 0.9*inch, 1.0*inch, 2.0*inch, 3.5*inch]
                table = Table(table_data, colWidths=col_widths, repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0,0), (-1,0), colors.grey),
                    ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                    ("ALIGN", (0,0), (2,-1), "CENTER"),
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ]))
                story.append(table)
                story.append(PageBreak())
                table_data = [["S.No", "Severity", "Image", "Key Findings", "Explanation"]]

        # --- Add remaining rows ---
        if len(table_data) > 1:
            col_widths = [0.6*inch, 0.9*inch, 1.0*inch, 2.0*inch, 3.5*inch]
            table = Table(table_data, colWidths=col_widths, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.grey),
                ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
                ("ALIGN", (0,0), (2,-1), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("GRID", (0,0), (-1,-1), 0.5, colors.black),
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ]))
            story.append(table)
    else:
        story.append(Paragraph("No hazards detected in uploaded images.", styles["Normal"]))

    # --- Build PDF with header/footer ---
    # If client requested a light/preview response (small payload), return compact JSON instead of full PDF
    light_mode = request.form.get("light", "false").lower() == "true"
    if light_mode:
        light_out = {
            "company": company_name,
            "site": site_name,
            "audit_date": audit_date,
            "summary": summary_text,
            "total_findings": len(all_findings),
            "findings": []
        }

        # include only small annotated thumbnails (first 6) to keep payload small
        for idx, row in enumerate(all_findings, start=1):
            sno, severity, img_idx, heading, explanation = row
            small_thumb = None
            if img_idx in annotated_thumbnails and idx <= 6:
                try:
                    annotated_thumbnails[img_idx].seek(0)
                    b = annotated_thumbnails[img_idx].getvalue()
                    small_thumb = base64.b64encode(b).decode("utf-8")
                except Exception:
                    small_thumb = None

            one_liner = None
            try:
                # one_liners may be set if all_findings branch ran
                one_liner = one_liners[idx-1] if idx-1 < len(one_liners) else (explanation.split(".")[0] + ".")
            except Exception:
                one_liner = (explanation.split(".")[0] + ".") if explanation else heading

            light_out["findings"].append({
                "sno": sno,
                "severity": severity,
                "heading": heading,
                "one_liner": one_liner,
                "thumbnail_b64": small_thumb
            })

        return jsonify(light_out)
    doc.build(
        story,
        onFirstPage=lambda c, d: add_header_footer(c, d, company_name, site_name, logo_stream),
        onLaterPages=lambda c, d: add_header_footer(c, d, company_name, site_name, logo_stream)
    )

    buff.seek(0)
    if request.form.get("preview", "false").lower() == "true":
        return send_file(buff, mimetype="application/pdf")

    return send_file(
        buff,
        as_attachment=True,
        download_name=f"{site_name}_audit_report.pdf",
        mimetype="application/pdf"
    )


@app.route("/report.html")
@login_required
def report_page():
    return send_file("template/report.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv("PORT", "5000"), debug=True)
