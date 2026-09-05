import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import spaces
from fpdf import FPDF
from PIL import Image

from cam import run_cam
from model_loader import INFER_TF, get_device, predict_tensor
from preprocess import smart_clean


def _quiet_api_info(self):
    return {"named_endpoints": {}, "unnamed_endpoints": {}}


gr.Blocks.get_api_info = _quiet_api_info

RESEARCHER = {
    "name": "Abu Saim Hossen Hridoy",
    "dept": "Department of CSE",
    "org": "Dhaka International University",
}

MESSAGES = {
    "Normal Person":
        "This ECG looks mostly normal. That is a good sign, but this app cannot replace a doctor. If you feel pain or unusual symptoms, please see a clinician.",
    "Abnormal heartbeat":
        "This ECG may show an irregular heartbeat. Try to stay calm. Please show this ECG to a doctor for a proper check.",
    "Myocardial Infarction":
        "This ECG may match a heart-attack pattern. This is not a confirmed diagnosis. If you have chest pain, sweating, or trouble breathing, get emergency help now.",
    "Patients that have History of MI":
        "This ECG may show signs of an old heart attack. Keep your regular check-ups and medicine. If new symptoms start, do not wait.",
}

FEATURES = {
    "Normal Person":
        "The model looked at the overall rhythm and wave shape. Bright areas on the highlight map are where it focused.",
    "Abnormal heartbeat":
        "The model looked for uneven beats and unusual wave shapes. Bright areas on the highlight map are where it focused.",
    "Myocardial Infarction":
        "The model looked for changes often seen in a heart attack, such as ST or Q-wave patterns. Bright areas show where it focused. This is not an exact lead report.",
    "Patients that have History of MI":
        "The model looked for lasting changes that can remain after an old heart attack. Bright areas show where it focused.",
}

TEAL = (11, 107, 88)
DARK = (17, 17, 17)
GRAY = (80, 80, 80)
LINE = (27, 92, 85)


def _save_rgb(src, dest: Path):
    if isinstance(src, (str, Path)):
        img = Image.open(src).convert("RGB")
    else:
        img = src.convert("RGB")
    img.save(dest)


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8, "Research prototype - Not a medical diagnosis - Page " + str(self.page_no()), align="C")


def make_pdf(patient, label, pct, message, why, probs, original, cleaned, cam):
    tmp = Path(tempfile.mkdtemp())
    orig_p = tmp / "original.png"
    clean_p = tmp / "cleaned.png"
    cam_p = tmp / "cam.png"
    _save_rgb(original, orig_p)
    _save_rgb(cleaned, clean_p)
    _save_rgb(cam, cam_p)

    pdf = ReportPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_text_color(*DARK)

    left, right, mid = 12, 198, 118

    pdf.set_xy(left, 16)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(100, 8, "ECG Screening Report")

    pdf.set_xy(mid, 16)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(80, 5, RESEARCHER["name"], align="R")
    pdf.set_xy(mid, 21)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(80, 5, RESEARCHER["dept"], align="R")
    pdf.set_xy(mid, 26)
    pdf.cell(80, 5, RESEARCHER["org"], align="R")
    pdf.set_xy(mid, 31)
    pdf.set_text_color(*GRAY)
    pdf.cell(80, 5, "Research prototype", align="R")
    pdf.set_text_color(*DARK)

    pdf.set_xy(left, 24)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(22, 6, "Patient:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(70, 6, str(patient)[:40])
    pdf.set_xy(left, 30)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(22, 6, "Date:")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(70, 6, datetime.now().strftime("%Y-%m-%d %H:%M"))

    pdf.set_draw_color(*LINE)
    pdf.set_line_width(0.6)
    pdf.line(left, 40, right, 40)

    pdf.set_xy(left, 46)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GRAY)
    pdf.cell(40, 5, "IMPRESSION")
    pdf.set_text_color(*DARK)

    pdf.set_fill_color(*TEAL)
    pdf.rect(left, 54, 2.2, 11, "F")
    pdf.set_xy(left + 5, 54)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(130, 11, str(label)[:42])
    pdf.set_xy(right - 36, 54)
    pdf.set_text_color(*TEAL)
    pdf.cell(36, 11, f"{pct:.1f}%", align="R")
    pdf.set_text_color(*DARK)

    pdf.set_xy(left, 70)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(right - left, 6, message)

    pdf.ln(2)
    pdf.set_x(left)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(right - left, 6, "Why this result: " + why)

    pdf.set_x(left)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(right - left, 5, f"Confidence {pct:.1f}% - screening only, not a medical diagnosis.")
    pdf.set_text_color(*DARK)
    pdf.ln(2)

    for name, p in probs.items():
        pdf.set_x(left)
        pdf.set_font("Helvetica", "B" if name == label else "", 11)
        if name != label:
            pdf.set_text_color(*GRAY)
        pdf.cell(150, 7, name)
        pdf.cell(36, 7, f"{p * 100:.1f}%", align="R")
        pdf.ln(7)
        pdf.set_text_color(*DARK)

    pdf.ln(6)
    y = pdf.get_y()
    img_w, gap = 58, 6
    captions = ["Original image", "Processed trace", "Highlight map"]
    paths = [orig_p, clean_p, cam_p]
    for i, (path, cap) in enumerate(zip(paths, captions)):
        with Image.open(path) as im:
            h = img_w * im.height / max(im.width, 1)
        x = left + i * (img_w + gap)
        pdf.image(str(path), x=x, y=y, w=img_w)
        pdf.set_xy(x, y + h + 2)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*GRAY)
        pdf.cell(img_w, 5, cap, align="C")
    pdf.set_text_color(*DARK)

    safe = "".join(ch for ch in str(patient) if ch.isalnum() or ch in " -_")
    safe = safe.strip().replace(" ", "_") or "Not_given"
    out = tmp / f"{safe}_ECG_report.pdf"
    pdf.output(str(out))
    return str(out)


def result_html(label, pct, msg, why, probs):
    bars = []
    for name, p in probs.items():
        active = " active" if name == label else ""
        bars.append(
            f'<div class="bar{active}"><div class="bar-top"><span>{name}</span>'
            f"<span>{p * 100:.1f}%</span></div>"
            f'<i><b style="width:{max(p * 100, 1):.1f}%"></b></i></div>'
        )
    return f"""
    <div class="report">
      <div class="verdict">
        <div>
          <div class="kicker">Impression</div>
          <h2>{label}</h2>
        </div>
        <div class="pct">{pct:.1f}%</div>
      </div>
      <p class="msg">{msg}</p>
      <p class="feat"><b>Why this result:</b> {why}</p>
      <p class="note">Confidence {pct:.1f}% - screening only, not a medical diagnosis.</p>
      <div class="bars">{''.join(bars)}</div>
    </div>
    """


def _analyze(image_path, source, patient):
    if image_path is None:
        raise gr.Error("Please upload an ECG image.")

    src = source or "original"
    if str(src).startswith("Full"):
        src = "original"
    elif str(src).startswith("Cropped"):
        src = "cropped"

    img = Image.open(image_path).convert("RGB")
    cleaned = smart_clean(img, src)
    result = predict_tensor(cleaned)

    device = get_device()
    tensor = INFER_TF(cleaned).unsqueeze(0).to(device)
    tensor = tensor.clone().detach().requires_grad_(True)
    overlay = run_cam(cleaned, tensor, result["pred_index"])

    label = result["pred_label"]
    pct = result["probabilities"][label] * 100
    msg = MESSAGES.get(label, "")
    why = FEATURES.get(label, "")
    name = (str(patient).strip() if patient else "") or "Not given"

    html = result_html(label, pct, msg, why, result["probabilities"])
    try:
        pdf_path = make_pdf(
            name, label, pct, msg, why, result["probabilities"],
            image_path, cleaned, overlay,
        )
    except Exception as pdf_err:
        html += f"<p class='note'>PDF could not be created: {pdf_err}</p>"
        pdf_path = None
    return html, cleaned, overlay, pdf_path, gr.update(visible=True)


@spaces.GPU(duration=60)
def _analyze_gpu(image_path, source, patient):
    return _analyze(image_path, source, patient)


def analyze(image_path, source, patient):
    try:
        return _analyze_gpu(image_path, source, patient)
    except Exception as exc:
        text = str(exc).lower()
        quota = any(
            word in text
            for word in ("quota", "zerogpu", "overquota", "gpu limit", "no gpu")
        )
        if quota:
            return _analyze(image_path, source, patient)
        raise gr.Error(f"{type(exc).__name__}: {exc}") from exc


CSS = """
@import url("https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@500&display=swap");

:root, html[data-ecg="dark"], .dark {
  --page: #07131a;
  --card: #0c202a;
  --text: #ffffff;
  --muted: #d0eee6;
  --head: #ffffff;
  --line: #2a5560;
  --input: #082028;
  --barbg: #1b3d46;
  --foot: #d0eee6;
}
html[data-ecg="light"] {
  --page: #e8f2ef;
  --card: #ffffff;
  --text: #0b1c20;
  --muted: #2f4d48;
  --head: #0b1c20;
  --line: #b7cfc8;
  --input: #ffffff;
  --barbg: #d5e8e3;
  --foot: #2f4d48;
}

body, .gradio-container, .main, .wrap, .contain, .app {
  background: var(--page) !important;
  color: var(--text) !important;
  font-family: "Source Sans 3", system-ui, sans-serif !important;
}

.gradio-container {
  max-width: 880px !important;
  width: 100% !important;
  margin: 0 auto !important;
  padding: 18px 16px 48px !important;
  position: relative !important;
  z-index: 1;
}

body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  opacity: 0.22;
  background-image:
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='48' viewBox='0 0 160 48'%3E%3Cpath fill='none' stroke='%2320c4a0' stroke-width='1.4' d='M0 24h22l4-10 6 28 8-36 6 24h18l3-8 5 16 4-12h84'/%3E%3C/svg%3E"),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='48' viewBox='0 0 180 48'%3E%3Cpath fill='none' stroke='%2320c4a0' stroke-width='1.2' d='M0 26h18l5-12 8 30 10-38 7 26h22l4-9 6 18 5-14h85'/%3E%3C/svg%3E"),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='150' height='48' viewBox='0 0 150 48'%3E%3Cpath fill='none' stroke='%2320c4a0' stroke-width='1.3' d='M0 22h16l4-8 7 24 9-32 6 22h16l3-7 5 14 4-10h66'/%3E%3C/svg%3E"),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='170' height='48' viewBox='0 0 170 48'%3E%3Cpath fill='none' stroke='%2320c4a0' stroke-width='1.1' d='M0 24h20l5-11 7 28 9-34 6 24h20l3-8 6 16 4-12h80'/%3E%3C/svg%3E");
  background-repeat: repeat-x;
  background-position: center 70px, center 260px, center 460px, center 680px;
}

.block, .form, .panel {
  position: relative !important;
  z-index: 2;
  background: var(--card) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  color: var(--text) !important;
}

.tophead { display: flex; align-items: center; gap: 12px; margin: 2px 0 14px; width: 100%; }
.logo {
  width: 44px; height: 44px; border-radius: 14px; flex: 0 0 44px;
  display: grid; place-items: center;
  background: #102226; border: 1px solid #1e4450;
  color: #e35d6a; font-size: 22px; font-weight: 700;
}
.tophead small {
  display: block; color: var(--muted); letter-spacing: .14em;
  text-transform: uppercase; font-size: 11px;
}
.tophead h1 { margin: 0; color: var(--head); font-size: 26px; }
.hb { flex: 1 1 auto; height: 42px; min-width: 60px; }
.hb svg { width: 100%; height: 42px; display: block; }
.theme-btns { display: flex; gap: 6px; flex: 0 0 auto; }
.theme-btns button {
  border: 1px solid var(--line) !important;
  background: var(--input) !important;
  color: var(--text) !important;
  border-radius: 999px !important;
  width: 32px !important;
  height: 32px !important;
  min-height: 32px !important;
  padding: 0 !important;
  display: grid !important;
  place-items: center !important;
  cursor: pointer;
}
.hint {
  margin: 4px 0 14px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid #20c4a0;
  background: rgba(32, 196, 160, 0.16);
  color: var(--text) !important;
  font-size: 14px;
  font-weight: 700;
  line-height: 1.45;
}
html[data-ecg="dark"] #ecg-dark,
html[data-ecg="light"] #ecg-light {
  background: #20c4a0 !important;
  color: #04241e !important;
  border-color: #20c4a0 !important;
}

.report {
  background: #0c202a !important;
  color: #e7f4f2 !important;
  padding: 14px 12px;
  border-radius: 12px;
}
.report, .report p, .report h2, .report span, .report b { color: #e7f4f2 !important; }
.report .kicker, .report .note, .report .bar-top { color: #8eada8 !important; }
.report .pct { color: #20c4a0 !important; }
.report .bar.active .bar-top { color: #e7f4f2 !important; }
.report .bar i { background: #1b3d46 !important; }

.verdict { display: flex; justify-content: space-between; gap: 12px; align-items: flex-end; }
.verdict h2 { margin: 0; font-size: clamp(22px, 5vw, 30px); }
.kicker { letter-spacing: .12em; text-transform: uppercase; font-size: 11px; }
.pct { font-family: "IBM Plex Mono", monospace; font-size: 28px; }
.msg, .feat { line-height: 1.55; margin: 12px 0 0; }
.note { font-size: 13px; margin-top: 8px; }
.bars { display: grid; gap: 9px; margin-top: 16px; }
.bar-top { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
.bar.active .bar-top { font-weight: 700; }
.bar i { display: block; height: 7px; border-radius: 99px; overflow: hidden; }
.bar i > b { display: block; height: 100%; background: linear-gradient(90deg, #159881, #20c4a0); }

label, .label-wrap span { color: var(--muted) !important; }
input, textarea, select, .wrap-inner {
  background: var(--input) !important;
  color: var(--text) !important;
  border-color: var(--line) !important;
  font-size: 16px !important;
}
button.primary {
  background: #20c4a0 !important;
  color: #04241e !important;
  font-weight: 700 !important;
  min-height: 44px !important;
}
img { max-width: 100% !important; height: auto !important; }
#ecg-upload { min-height: 300px !important; }
#ecg-upload .image-container,
#ecg-upload .image-frame,
#ecg-upload .upload-container,
#ecg-upload .wrap,
#ecg-upload .column {
  width: 100% !important;
  max-width: 100% !important;
}
#ecg-upload img,
#ecg-upload video {
  width: 100% !important;
  max-height: 240px !important;
  object-fit: contain !important;
}
.foot { color: var(--foot); font-size: 13px; margin-top: 18px; line-height: 1.5; }

footer {
  display: none !important;
}

#pdf-download {
  margin: 10px 0 14px !important;
}
#pdf-download label,
#pdf-download .label-wrap {
  display: none !important;
}
#pdf-download button {
  width: 100% !important;
  min-height: 48px !important;
  border-radius: 12px !important;
  background: #20c4a0 !important;
  color: #04241e !important;
  font-weight: 700 !important;
  font-size: 16px !important;
}

@media (max-width: 800px) {
  .form, .row { flex-direction: column !important; }
  .form > *, .row > * { width: 100% !important; min-width: 0 !important; flex: 1 1 100% !important; }
  .verdict { display: block; }
}
"""

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Source Sans 3"),
).set(
    button_primary_background_fill="#20c4a0",
    button_primary_text_color="#04241e",
)

with gr.Blocks(title="ECG Classifier", theme=theme, css=CSS) as demo:
    gr.HTML(
        """
        <div class="tophead">
          <div class="logo">+</div>
          <div>
            <small>Cardiac screening</small>
            <h1>ECG Classifier</h1>
          </div>
          <div class="hb" aria-hidden="true">
            <svg viewBox="0 0 320 48" preserveAspectRatio="none">
              <path fill="none" stroke="#20c4a0" stroke-width="2"
                d="M0 24h36l6-14 10 34 12-40 8 28h28l5-10 8 20 6-16h36l6-14 10 34 12-40 8 28h28l5-10 8 20 6-16h70"/>
            </svg>
          </div>
          <div class="theme-btns">
            <button type="button" id="ecg-dark" title="Dark" aria-label="Dark">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true">
                <path d="M21 14.3A8.5 8.5 0 0 1 9.7 3 8.6 8.6 0 1 0 21 14.3z"/>
              </svg>
            </button>
            <button type="button" id="ecg-light" title="Light" aria-label="Light">
              <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <circle cx="12" cy="12" r="4"/>
                <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>
              </svg>
            </button>
          </div>
        </div>
        """
    )

    image = gr.Image(
        type="filepath",
        label="Upload ECG",
        height=300,
        elem_id="ecg-upload",
        sources=["upload", "webcam", "clipboard"],
    )
    gr.HTML(
        """
        <div class="hint">
          <b>Select the correct Image type, or confidence will drop.</b><br>
          <b>Full ECG paper</b> - the whole hospital printout: patient name, hospital info, grid, and all leads.<br>
          <b>Cropped strip</b> - only one cut wave line, no hospital or patient header.
        </div>
        """
    )
    with gr.Row():
        source = gr.Dropdown(
            choices=["Full ECG paper", "Cropped strip"],
            value="Full ECG paper",
            label="Image type",
        )
        patient = gr.Textbox(label="Patient name", placeholder="Patient name")
        btn = gr.Button("Analyze ECG", variant="primary")

    with gr.Column(visible=False) as results:
        out_html = gr.HTML()
        with gr.Row():
            out_clean = gr.Image(label="Processed trace", height=220)
            out_cam = gr.Image(label="Highlight map", height=220)
        if hasattr(gr, "DownloadButton"):
            out_pdf = gr.DownloadButton(
                label="Download report",
                value=None,
                variant="primary",
                elem_id="pdf-download",
            )
        else:
            out_pdf = gr.File(label="Download report", elem_id="pdf-download")

    gr.HTML(
        f"""
        <div class="foot">
          Research prototype for academic use only. Not a medical diagnosis.<br>
          {RESEARCHER['name']} · {RESEARCHER['dept']}, {RESEARCHER['org']}
        </div>
        """
    )

    demo.load(
        fn=None,
        js="""
() => {
  const root = document.documentElement;
  const apply = (mode) => {
    const dark = mode === "dark";
    root.setAttribute("data-ecg", dark ? "dark" : "light");
    root.classList.toggle("dark", dark);
    document.body.classList.toggle("dark", dark);
    try { localStorage.setItem("ecg-theme", dark ? "dark" : "light"); } catch (e) {}
  };
  const saved = (() => { try { return localStorage.getItem("ecg-theme"); } catch (e) { return null; } })();
  apply(saved || (root.classList.contains("dark") ? "dark" : "dark"));
  const bind = () => {
    const d = document.getElementById("ecg-dark");
    const l = document.getElementById("ecg-light");
    if (d) d.onclick = () => apply("dark");
    if (l) l.onclick = () => apply("light");
  };
  bind();
  new MutationObserver(bind).observe(document.body, { childList: true, subtree: true });
}
""",
    )
    btn.click(
        analyze,
        inputs=[image, source, patient],
        outputs=[out_html, out_clean, out_cam, out_pdf, results],
    )

if __name__ == "__main__":
    try:
        demo.queue()
    except Exception:
        pass
    demo.launch()