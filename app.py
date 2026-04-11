"""
app.py — Drywall QA Prompted Segmentation
Model  : S-4-G-4-R/clipseg-drywall-qa
"""

# ── Force install torch if missing (HF Spaces fallback) ──────────────────────
import subprocess, sys

def install_torch():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.6.0+cpu",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--quiet"
    ])

try:
    import torch
except ModuleNotFoundError:
    print("torch not found — installing...")
    install_torch()
    import torch

import os
import time
import numpy as np
import torch
import gradio as gr
from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ── Config ────────────────────────────────────────────────────────────────────
REPO_ID   = "S-4-G-4-R/clipseg-drywall-qa"
THRESHOLD = 0.5
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_CHOICES = ["segment crack", "segment taping area"]

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading from {REPO_ID} on {DEVICE} ...")
processor = CLIPSegProcessor.from_pretrained(REPO_ID)
model     = CLIPSegForImageSegmentation.from_pretrained(REPO_ID)
model     = model.to(DEVICE)
model.eval()
print("Ready.")

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(image, prompt, threshold):
    if image is None:
        return None, None, "⚠  Upload an image to begin."

    original_size = image.size
    image_rgb     = image.convert("RGB")

    encoding = processor(
        text=prompt, images=image_rgb,
        return_tensors="pt", padding="max_length", truncation=True,
    )
    pixel_values   = encoding["pixel_values"].to(DEVICE)
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask)
    inf_ms = (time.time() - t0) * 1000

    prob     = torch.sigmoid(outputs.logits[0]).cpu().numpy()
    pred_bin = (prob > threshold).astype(np.uint8)
    mask_pil = Image.fromarray((pred_bin * 255).astype(np.uint8), mode="L")
    mask_pil = mask_pil.resize(original_size, Image.NEAREST)
    mask_arr = np.array(mask_pil)

    img_arr = np.array(image_rgb).astype(np.float32)
    overlay = img_arr.copy()
    colour  = np.array([0, 210, 230], dtype=np.float32) if "crack" in prompt \
              else np.array([255, 160, 50], dtype=np.float32)
    fg = mask_arr > 0
    overlay[fg] = overlay[fg] * 0.4 + colour * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    coverage = fg.sum() / fg.size * 100
    info = (
        f"Prompt     :  {prompt}\n"
        f"Threshold  :  {threshold:.2f}\n"
        f"Inference  :  {inf_ms:.1f} ms\n"
        f"Coverage   :  {coverage:.2f}%  of image\n"
        f"Device     :  {DEVICE}"
    )
    return Image.fromarray(overlay), mask_pil, info


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg-primary:   #0a0a0f;
    --bg-secondary: #111118;
    --bg-card:      #16161f;
    --bg-hover:     #1e1e2a;
    --border:       #2a2a3a;
    --border-bright:#3a3a55;
    --accent-cyan:  #00d4e8;
    --accent-orange:#ff9f43;
    --accent-purple:#a78bfa;
    --text-primary: #e8e8f0;
    --text-secondary:#8888aa;
    --text-muted:   #55556a;
    --success:      #10d982;
    --radius:       12px;
}
/* ── Base ── */
body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}
/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0a0a0f 0%, #12121e 50%, #0a0f1a 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 36px 40px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.header-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,212,232,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.header-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(167,139,250,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.header-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.5px;
    margin: 0 0 6px !important;
    line-height: 1.2 !important;
}
.header-title span {
    color: var(--accent-cyan);
}
.header-subtitle {
    font-size: 14px !important;
    color: var(--text-secondary) !important;
    font-weight: 300 !important;
    margin: 0 !important;
    letter-spacing: 0.3px;
}
.tag-row {
    display: flex;
    gap: 8px;
    margin-top: 18px;
    flex-wrap: wrap;
}
.tag {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    font-weight: 400;
}
.tag-cyan  { background: rgba(0,212,232,0.1);  color: var(--accent-cyan);   border: 1px solid rgba(0,212,232,0.2); }
.tag-orange{ background: rgba(255,159,67,0.1); color: var(--accent-orange); border: 1px solid rgba(255,159,67,0.2);}
.tag-purple{ background: rgba(167,139,250,0.1);color: var(--accent-purple); border: 1px solid rgba(167,139,250,0.2);}
/* ── Metric pills ── */
.metrics-row {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.metric-pill {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 20px;
    flex: 1;
    min-width: 160px;
    transition: border-color 0.2s;
}
.metric-pill:hover { border-color: var(--border-bright); }
.metric-pill .value {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-pill .label {
    font-size: 11px;
    color: var(--text-secondary);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.cyan-val   { color: var(--accent-cyan);   }
.orange-val { color: var(--accent-orange); }
.purple-val { color: var(--accent-purple); }
.green-val  { color: var(--success);       }
/* ── Panel cards ── */
.panel-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    height: 100%;
}
.panel-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
/* ── Gradio component overrides ── */
.gradio-container .block,
.gradio-container .form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
/* Image upload area */
.gradio-container .upload-container,
.gradio-container [data-testid="image"] {
    background: var(--bg-secondary) !important;
    border: 1.5px dashed var(--border-bright) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s !important;
}
.gradio-container [data-testid="image"]:hover {
    border-color: var(--accent-cyan) !important;
}
/* Radio buttons */
.gradio-container .wrap.svelte-1p9xokt,
.gradio-container .wrap {
    gap: 10px !important;
}
.gradio-container input[type="radio"] + span,
.gradio-container .radio-item {
    background: var(--bg-secondary) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    padding: 10px 16px !important;
    font-size: 13px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    font-family: 'Space Mono', monospace !important;
}
.gradio-container input[type="radio"]:checked + span {
    background: rgba(0,212,232,0.08) !important;
    border-color: var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
}
/* Slider */
.gradio-container input[type="range"] {
    accent-color: var(--accent-cyan) !important;
}
.gradio-container .slider-container {
    background: transparent !important;
}
/* Textbox output */
.gradio-container textarea,
.gradio-container .output-class {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.8 !important;
}
/* Labels */
.gradio-container label span,
.gradio-container .label-wrap span {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    font-family: 'Space Mono', monospace !important;
}
/* Run button */
.gradio-container button.primary {
    background: linear-gradient(135deg, var(--accent-cyan), #0098a8) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #0a0a0f !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(0,212,232,0.25) !important;
}
.gradio-container button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(0,212,232,0.4) !important;
}
.gradio-container button.primary:active {
    transform: translateY(0) !important;
}
/* Footer */
.footer-text {
    text-align: center;
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'Space Mono', monospace;
    padding: 20px 0 8px;
    letter-spacing: 0.5px;
}
.footer-text a {
    color: var(--accent-cyan);
    text-decoration: none;
}
/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
"""

# ── HTML blocks ───────────────────────────────────────────────────────────────
HEADER_HTML = """
<div class="header-banner">
  <div class="header-title">🧱 Drywall <span>QA</span> — Prompted Segmentation</div>
  <div class="header-subtitle">
    Fine-tuned CLIPSeg · Text-conditioned binary mask generation · Drywall defect detection
  </div>
  <div class="tag-row">
    <span class="tag tag-cyan">CLIPSeg</span>
    <span class="tag tag-orange">PyTorch</span>
    <span class="tag tag-purple">HuggingFace</span>
    <span class="tag tag-cyan">Seed 42</span>
    <span class="tag tag-orange">20 Epochs</span>
    <span class="tag tag-purple">352×352</span>
  </div>
</div>
"""

METRICS_HTML = """
<div class="metrics-row">
  <div class="metric-pill">
    <div class="value cyan-val">0.735</div>
    <div class="label">crack · val mIoU</div>
  </div>
  <div class="metric-pill">
    <div class="value green-val">0.834</div>
    <div class="label">crack · val dice</div>
  </div>
  <div class="metric-pill">
    <div class="value orange-val">0.499</div>
    <div class="label">taping · val mIoU</div>
  </div>
  <div class="metric-pill">
    <div class="value purple-val">0.626</div>
    <div class="label">taping · val dice</div>
  </div>
</div>
"""

FOOTER_HTML = """
<div class="footer-text">
  Model →
  <a href="https://huggingface.co/S-4-G-4-R/clipseg-drywall-qa" target="_blank">
    S-4-G-4-R/clipseg-drywall-qa
  </a>
  &nbsp;·&nbsp; Base: CIDAS/clipseg-rd64-refined &nbsp;·&nbsp; Datasets: Roboflow
</div>
"""

# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="Drywall QA Segmentation", theme=gr.themes.Base()) as demo:

    gr.HTML(HEADER_HTML)
    gr.HTML(METRICS_HTML)

    with gr.Row(equal_height=False):

        # Left — inputs
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="INPUT IMAGE",
                height=300,
            )
            prompt_input = gr.Radio(
                choices=PROMPT_CHOICES,
                value=PROMPT_CHOICES[0],
                label="SEGMENTATION PROMPT",
            )
            threshold_slider = gr.Slider(
                minimum=0.1, maximum=0.9,
                value=THRESHOLD, step=0.05,
                label="THRESHOLD",
            )
            run_btn = gr.Button("⟶  RUN SEGMENTATION", variant="primary")

        # Right — outputs
        with gr.Column(scale=1):
            overlay_out = gr.Image(
                type="pil",
                label="OVERLAY",
                height=300,
            )
            with gr.Row():
                mask_out = gr.Image(
                    type="pil",
                    label="BINARY MASK",
                    height=160,
                )
                info_out = gr.Textbox(
                    label="RUN INFO",
                    lines=7,
                )

    run_btn.click(
        fn=predict,
        inputs=[image_input, prompt_input, threshold_slider],
        outputs=[overlay_out, mask_out, info_out],
    )
    image_input.change(
        fn=predict,
        inputs=[image_input, prompt_input, threshold_slider],
        outputs=[overlay_out, mask_out, info_out],
    )

    gr.HTML(FOOTER_HTML)


if __name__ == "__main__":
    demo.launch()
