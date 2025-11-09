#Making a Gemini prompt file to call from app.py
import os

# ------------------------ Auditor Prompt ------------------------
def auditor(w, h) -> str:
    return f"""
        You are an expert safety auditor assistant.

        Task:
        Analyze the uploaded image ({w}x{h} pixels) and identify hazards that pose a risk to human life
        or violate general safety standards. Return ONLY a valid JSON object.

        JSON specification:
        {{
        "wrong": [
            {{
            "heading": "short title (≤5 words)",
            "severity": <integer 1–5>,
            "explanation": "1–3 sentences",
            "box": [x1, y1, x2, y2]
            }}
        ],
        "right": [
            {{ "heading": "short title" }}
        ],
        "todo": [
            {{ "heading": "short action point" }}
        ]
        }}

        Important rules:
        1. Do not return markdown, comments, or text outside the JSON.
        2. All bounding boxes MUST use absolute pixel coordinates.
        3. If no hazards are found, return "wrong": [].
        4. Return up to 10 items per section.
    """


# ------------------------ Single Line Summary Prompt ------------------------
def single_line_summary(context:list) -> str:
    return """
        You are an expert content generator.

        Task:
        Generate a concise single-line summary of the context provided in list.

        context: {context}
        Return ONLY the summary text.

        Important rules:
        1. Do not return markdown, comments, or text outside the summary.
        2. The summary must be a single sentence with a maximum of 10 words.
    """
    
# ------------------------ Multi Line Summary Prompt ------------------------
def multi_line_summary(context:list) -> str:
    return """
        You are an expert content generator.

        Task:
        Generate a concise multi-line summary of the context provided in list.

        context: {context}
        Return ONLY the summary text.

        Important rules:
        1. Do not return markdown, comments, or text outside the summary.
        2. The summary must be 5-7 sentences long.
    """

#------------------------ Good Findings Image Report Prompt ------------------------
def good_finds_image_report(w,h) -> str:
    return f"""
        You are an expert safety auditor assistant.

        Task:
        Analyze the uploaded image ({w}x{h} pixels) and identify good safety practices that ensure
        a safe working environment. Return ONLY the best good practice of the image shown.

        Important rules:
        1. Do not return markdown or comments.
        2. The good practice must be a single sentence with a maximum of 10-12 words.
    """
    