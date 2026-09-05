# ECG Classifier

Live demo: [Hugging Face Space](https://huggingface.co/spaces/Saimhosenhridoy/ECG-Scanning)

Research prototype for ECG image classification using **ConvNeXt-Tiny** and **Grad-CAM**.

**Author:** Abu Saim Hossen Hridoy  
**Affiliation:** Department of CSE, Dhaka International University

This project is for academic use only. **It is not a medical diagnosis** and cannot replace a clinician.

## Live demo

The app is deployed on Hugging Face Spaces:

https://huggingface.co/spaces/Saimhosenhridoy/ECG-Scanning

Open that link, upload an ECG image, select the image type, and click **Analyze ECG**.

No local install is needed for the demo.

## Features

- Upload, paste, or capture an ECG image
- Two image types: Full ECG paper / Cropped strip
- 4-class prediction with confidence
- Grad-CAM highlight map
- PDF screening report
- Dark / Light theme

## How to use

### On Hugging Face (recommended)

1. Open the Space link above.
2. Upload an ECG image (file, drag-drop, paste, or one webcam photo).
3. Select **Image type** before Analyze. Wrong type can lower confidence.
   - **Full ECG paper** — whole hospital printout: patient name, hospital info, grid, and all leads
   - **Cropped strip** — only one cut wave line, no hospital or patient header
4. Optional: type a patient name for the PDF. If empty, the report shows `Not given`.
5. Click **Analyze ECG**.
6. Read the impression, confidence, message, processed trace, and highlight map.
7. Download the PDF report.

Classes:

- Normal Person
- Abnormal heartbeat
- Myocardial Infarction
- Patients that have History of MI

Dark / Light: moon button = Dark, sun button = Light.

The first run on the Space may take longer while the model loads.

### Local run

Use this only if you want to run the same app on your computer.

    git clone https://github.com/Saimhosenhridoy/ECG-Scanning.git
    cd ECG-Scanning

Windows:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python app.py

Linux / macOS:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python app.py

Then open http://127.0.0.1:7860

Put the trained weights here first:

`weights/convnext_tiny_ecg.pth`

## Project structure

    ECG-Scanning/
    ├── app.py
    ├── cam.py
    ├── model_loader.py
    ├── preprocess.py
    ├── settings.py
    ├── requirements.txt
    ├── weights/
    │   └── convnext_tiny_ecg.pth
    └── README.md

## Requirements

- Python 3.10+
- Model file in `weights/` for local run
- Hugging Face Space hosts the public demo

## Disclaimer

This is a research prototype. Results can be wrong.

Do not use it for emergency or clinical decisions.

If there is chest pain, sweating, or trouble breathing, get emergency help.

## License

Academic / research use.

Source: [GitHub](https://github.com/Saimhosenhridoy/ECG-Scanning)  
Demo: [Hugging Face Space](https://huggingface.co/spaces/Saimhosenhridoy/ECG-Scanning)