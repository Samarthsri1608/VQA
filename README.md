🦺 Safety Vision

AI-Powered Visual Safety Auditor

📘 Overview

Safety Vision is an AI-based web tool that automatically detects safety hazards from uploaded workplace or factory images.
It uses Google Gemini Vision (via google-genai) to analyze an image, identify unsafe conditions, and categorize observations as:

Wrong: Unsafe or hazardous conditions

Right: Correct safety measures in place

To-Do: Recommended corrective actions

Each hazard is visually highlighted on the image with bounding boxes and severity indicators.

🧩 Key Features

✅ Upload workplace or environment images for instant safety assessment
✅ AI-driven hazard detection using Gemini 2.5 Flash
✅ Auto-generated bounding boxes with labels and confidence levels
✅ Clear categorization: Wrong, Right, and To-Do
✅ Interactive frontend with expandable accordion sections
✅ Fully local deployment using Flask + HTML/JS

🗂️ Project Structure
SafetyVision/
│
├── app.py               # Flask backend with Gemini API integration
├── index.html           # Frontend UI for upload and results
├── requirements.txt     # Python dependencies (see below)
└── README.md            # Documentation

⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/yourusername/safety-vision.git
cd safety-vision

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # (on Linux/Mac)
venv\Scripts\activate      # (on Windows)

3. Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt, create one with:

Flask
flask-cors
Pillow
numpy
python-dotenv
google-genai

4. Set up your environment variables

Create a .env file in the project root with:

GEMINI_API_KEY=your_google_api_key_here

5. Run the app
python app.py


Visit 👉 http://localhost:5000
 in your browser.

🧠 How It Works

Image Upload:
You upload an image (JPG/PNG/etc.) of a workplace or site.

Gemini Vision Analysis:
The backend sends the image + a structured JSON prompt to Gemini 2.5 Flash.

AI Response Parsing:
Gemini returns a JSON with hazard detections, severity, and bounding boxes.

Visualization:
Detected hazards are drawn on the image using Pillow and displayed with severity labels.

Frontend Display:
The web interface dynamically lists “Wrong,” “Right,” and “To-Do” findings in collapsible sections.

🧰 Tech Stack
Layer	Technology
Backend	Flask, Python, google-genai, Pillow
Frontend	HTML, CSS, JavaScript
AI Model	Gemini 2.5 Flash (Vision + Text)
Styling	Custom dark-themed responsive layout
🧪 Example Workflow

Upload an image of a workshop or site.

Click “Run Analysis.”

The app sends it to Gemini and receives structured results.

View hazards highlighted in red boxes on the annotated image.

Review the detailed findings in the accordion panels.

⚠️ Notes

Ensure your Google API key has access to Gemini Vision models.

This project is intended for educational and prototype use only — not as a certified safety audit tool.

🏗️ Future Enhancements

Multi-image batch processing

PDF report export

Role-based dashboards (auditor, company, site level)

Daily “Quick Safety Check” mode via camera feed

👨‍💻 Author

Samarth Srivastava
Project under the VisionIQ initiative.