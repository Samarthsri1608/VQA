import io
import datetime
from flask import Blueprint, request, jsonify, send_file, current_app
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

report_bp = Blueprint("report", __name__)

@report_bp.route("/multi_report", methods=["POST"])
def multi_report():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    site_location = request.form.get("location", "Not specified")
    audit_date = request.form.get("date", str(datetime.date))
    images = request.files.getlist("images")

    all_findings = []
    summary_points = []

    # Use existing /detect route from app.py (Flask internal call)
    with current_app.test_client() as client:
        for idx, f in enumerate(images, start=1):
            fd = {"image": (f, f.filename)}
            res = client.post("/detect", data=fd, content_type="multipart/form-data")
            data = res.get_json() or {}

            for w_item in data.get("wrong", []):
                all_findings.append([
                    str(idx),
                    str(w_item.get("severity", "")),
                    f"Image {idx}",
                    w_item.get("heading", ""),
                    w_item.get("explanation", "")
                ])
                summary_points.append(w_item.get("heading", ""))

    # --- PDF Generation ---
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    story.append(Paragraph("<b>SAFETY AUDIT REPORT</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Location of Site: {site_location}", styles["Normal"]))
    story.append(Paragraph(f"Audit Date: {audit_date}", styles["Normal"]))
    story.append(Paragraph(f"Report Generated: {datetime.datetime.now()}", styles["Normal"]))
    story.append(Spacer(1, 12))

    summary_text = "Summary: " + (", ".join(summary_points[:10]) or "No major issues detected.")
    story.append(Paragraph(summary_text, styles["Normal"]))
    story.append(Spacer(1, 12))

    if all_findings:
        data = [["S.No", "Severity", "Image", "Key Findings", "Explanation"]] + all_findings
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No hazards detected in uploaded images.", styles["Normal"]))

    doc.build(story)
    buff.seek(0)

    return send_file(buff, as_attachment=True,
                     download_name="safety_report.pdf",
                     mimetype="application/pdf")
