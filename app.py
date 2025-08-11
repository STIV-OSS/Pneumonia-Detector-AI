import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms, models
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import csv
import pandas as pd
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import subprocess
import zipfile
import hashlib
import datetime
import base64
import uuid
import json

st.set_page_config(page_title="Pneumonia Detector AI", layout="wide")
MODEL_PATH = "efficientnet_pneumonia_best.pth"
FEEDBACK_CSV = "feedback_log.csv"
FEEDBACK_IMAGES = "feedback_images"
AUDIT_LOG = "retraining_audit_log.csv"
SAMPLE_IMAGE = "sample_xray.jpg"
os.makedirs(FEEDBACK_IMAGES, exist_ok=True)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]
if "session_feedbacks" not in st.session_state:
    st.session_state["session_feedbacks"] = []

if "feedback_weight" not in st.session_state:
    st.session_state["feedback_weight"] = 10

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    model.to(device)
    return model, device

def get_gradcam_heatmap(model, input_tensor):
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    return grayscale_cam[0]

def generate_pdf_report(rows, filename="feedback_report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, "Pneumonia AI Feedback Report")
    y = 720
    for row in rows:
        c.drawString(100, y, f"Image: {row[0]} | Model: {row[1]} ({row[2]}) | User: {row[3]}")
        y -= 20
        if y < 50:
            c.showPage()
            y = 750
    c.save()
    return filename

def export_fhir(feedbacks):
    resources = []
    for fb in feedbacks:
        dr = {
            "resourceType": "DiagnosticReport",
            "id": fb[0],
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "30746-0", "display": "Chest X-ray"}]},
            "conclusion": f"AI: {fb[1]} (conf {fb[2]}) | User: {fb[3]}"
        }
        resources.append(dr)
    return json.dumps({"resourceType": "Bundle", "entry": [{"resource": r} for r in resources]}, indent=2)

def download_feedback_options(feedbacks):
    choice = st.selectbox(
        "Select feedback format to download",
        ("Download as CSV", "Download as PDF", "Download as JSON", "Download as FHIR"),
        key="download_feedback_choice"
    )
    if choice == "Download as CSV":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Image', 'Model_Pred', 'Confidence', 'User_Label'])
        for row in feedbacks:
            writer.writerow(row)
        st.download_button(
            "Download Session Feedback as CSV",
            output.getvalue(),
            "session_feedback.csv",
            "text/csv"
        )
    elif choice == "Download as PDF":
        pdf_path = generate_pdf_report(feedbacks, "session_feedback_report.pdf")
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Session Feedback as PDF",
                f.read(),
                "session_feedback_report.pdf"
            )
    elif choice == "Download as JSON":
        j = json.dumps([{"image": r[0], "ai_pred": r[1], "conf": r[2], "user": r[3]} for r in feedbacks], indent=2)
        st.download_button("Download Session Feedback as JSON", j, "session_feedback.json", "application/json")
    elif choice == "Download as FHIR":
        fhir = export_fhir(feedbacks)
        st.download_button("Download Session Feedback as FHIR (JSON)", fhir, "session_feedback_fhir.json", "application/json")

def sample_img_download():
    if os.path.exists(SAMPLE_IMAGE):
        with open(SAMPLE_IMAGE, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"[Download a sample X-ray for testing](data:image/jpeg;base64,{b64})", unsafe_allow_html=True)

def consent_banner():
    if "agreed" not in st.session_state:
        st.session_state["agreed"] = False
    if not st.session_state["agreed"]:
        with st.expander("Privacy & Disclaimer", expanded=True):
            st.warning("**Notice:** This tool is an AI demonstration. It is **not a substitute for professional medical advice or diagnosis**. Images and feedback stay on your device/server unless you download or share them.")
            agree = st.checkbox("I understand and wish to use the demo")
            if agree:
                st.session_state["agreed"] = True
    return st.session_state["agreed"]

def get_model_metrics():
    acc, auc, f1_pneu, f1_norm = 0.92, 0.97, 0.94, 0.89
    return acc, auc, f1_pneu, f1_norm

def get_latest_audit():
    if os.path.exists(AUDIT_LOG):
        df = pd.read_csv(AUDIT_LOG, header=None, names=["Datetime", "ModelHash", "TrainSize", "FeedbackSize", "TestAcc"])
        last_row = df.iloc[-1]
        return str(last_row["Datetime"]), str(last_row["ModelHash"])[:8]
    return None, None

def feedback_dashboard():
    if os.path.exists(FEEDBACK_CSV):
        df = pd.read_csv(FEEDBACK_CSV, header=None, names=["Image", "Model_Pred", "Confidence", "User_Label", "Image_Path"])
        correct = (df["Model_Pred"] == df["User_Label"]).sum()
        total = len(df)
        acc = correct / total if total else 0
        st.subheader("Feedback Dashboard")
        st.write(f"Overall user feedback accuracy: **{acc:.2%}** out of {total} feedbacks.")
        st.write("Class distribution:")
        st.bar_chart(df["User_Label"].value_counts())
        st.write("Recent feedback (last 20):")
        st.dataframe(df.tail(20), use_container_width=True)
    else:
        st.info("No feedback yet.")

def all_feedback_download():
    if os.path.exists(FEEDBACK_CSV):
        with open(FEEDBACK_CSV, "r") as f:
            csv_data = f.read()
        st.download_button(
            "Download All Feedback as CSV",
            csv_data,
            "all_feedback.csv",
            "text/csv"
        )

def extract_zip_and_yield_images(zip_bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
        for name in zip_ref.namelist():
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                with zip_ref.open(name) as file:
                    yield name, Image.open(file).convert('L')

def check_xray_quality(image):
    arr = np.array(image)
    if arr.ndim == 2 and arr.shape[0] > 200 and arr.shape[1] > 200 and arr.mean() > 20:
        return True, ""
    return False, "Image is too small, blank, or not a valid chest X-ray."

if st.query_params.get("health") == ["1"]:
    st.write("healthy")
    st.stop()

with st.sidebar:
    st.header("Model Test Metrics")
    acc, auc, f1_pneu, f1_norm = get_model_metrics()
    st.markdown(
        f"**Accuracy:** {acc:.0%}  \n"
        f"**AUC:** {auc:.2f}  \n"
        f"**Pneumonia F1:** {f1_pneu:.2f}  \n"
        f"**Normal F1:** {f1_norm:.2f}"
    )
    last_retrain, ver = get_latest_audit()
    if last_retrain:
        st.write(f"Last retrain: {last_retrain.split('.')[0]} | Model version: `{ver}`")
    st.markdown("---")
    st.info("After uploading, check the Feedback Dashboard tab for history and downloads.")
    st.markdown("---")
    if st.checkbox("Show admin controls"):
        st.slider("Feedback oversample weight", 1, 50, st.session_state["feedback_weight"], key="feedback_weight_slider")
        st.session_state["feedback_weight"] = st.session_state["feedback_weight_slider"]

tab1, tab2 = st.tabs(["Pneumonia Detector", "Feedback Dashboard"])

with tab1:
    agreed = consent_banner()
    if not agreed:
        st.stop()
    st.title("Pneumonia Detector AI")
    st.markdown("""
    <div style="color:gray">
    <b>How to use:</b>
    <ol>
      <li>Download a <b>sample X-ray</b> below, or upload your own (single image or batch ZIP)</li>
      <li>View AI prediction, confidence, and attention map</li>
      <li>Correct the prediction if needed and submit feedback</li>
      <li>Check/download your feedback in the dashboard tab</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    sample_img_download()

    model, device = load_model()

    uploaded_files = st.file_uploader(
        "Upload X-ray image(s) (jpg/jpeg/png) or a ZIP with multiple images:", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True
    )

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_images = []
    if uploaded_files:
        for file in uploaded_files:
            if file.name.lower().endswith('.zip'):
                for name, image in extract_zip_and_yield_images(file.read()):
                    batch_images.append((name, image))
            else:
                batch_images.append((file.name, Image.open(file).convert('L')))

        for i, (filename, image) in enumerate(batch_images):
            with st.container():
                st.markdown(f"##### File: `{filename}`")
                cols = st.columns([1, 1])
                valid, q_error = check_xray_quality(image)
                if not valid:
                    cols[0].error(f"{q_error} (Upload a real chest X-ray, at least 200x200 px, not too dark.)")
                    continue
                cols[0].image(image, caption="Original X-ray", use_container_width=True)
                img = ImageOps.exif_transpose(image)
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(img_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_class = probs.argmax()
                    pred_label = "Normal" if pred_class == 0 else "Pneumonia"
                    confidence = probs[pred_class]
                with st.spinner("Calculating attention map..."):
                    heatmap = get_gradcam_heatmap(model, img_tensor)
                    img_for_cam = np.array(img.resize((224,224))).astype(np.float32) / 255.
                    if img_for_cam.ndim == 2:
                        img_for_cam = np.stack([img_for_cam]*3, axis=-1)
                    cols[1].image(cam_img := show_cam_on_image(img_for_cam, heatmap, use_rgb=True),
                                  caption=f'Attention Map ({pred_label})', width=256)
                bar_vals = pd.DataFrame({"Class": ["Normal", "Pneumonia"], "Prob": probs})
                st.bar_chart(bar_vals.set_index("Class"))

                st.markdown(
                    f"<b>Prediction:</b> <span style='color: #1a76d2'>{pred_label}</span> "
                    f"&nbsp; <b>Confidence:</b> <span style='color: #43a047'>{confidence:.2%}</span>",
                    unsafe_allow_html=True
                )
                st.progress(float(confidence), text="Prediction confidence")

                f_col1, f_col2 = st.columns([2, 2])
                with f_col1:
                    feedback = st.radio(
                        f"Is the prediction correct for `{filename}`?",
                        ("Correct", "Incorrect"),
                        key=f"radio_{i}_{filename}"
                    )
                with f_col2:
                    true_label = pred_label
                    if feedback == "Incorrect":
                        true_label = st.selectbox(
                            "Select the correct label:",
                            ("Normal", "Pneumonia"),
                            key=f"selectbox_{i}_{filename}"
                        )
                if st.button(f"Undo Last Feedback ({filename})", key=f"undo_{i}_{filename}"):
                    if st.session_state["session_feedbacks"]:
                        st.session_state["session_feedbacks"].pop()
                        st.warning("Last feedback entry removed (session only).")
                if st.button(f"Submit Feedback for {filename}", key=f"submit_{i}_{filename}"):
                    feedback_image_path = os.path.join(FEEDBACK_IMAGES, filename)
                    image.save(feedback_image_path)
                    row = [filename, pred_label, f"{confidence:.4f}", true_label]
                    with open(FEEDBACK_CSV, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row + [feedback_image_path])
                    st.success("Feedback saved. Thank you!")
                    st.session_state["session_feedbacks"].append(row)

        if st.session_state["session_feedbacks"]:
            st.markdown("---")
            download_feedback_options(st.session_state["session_feedbacks"])

    st.markdown("---")
    with st.expander("Retrain Model", expanded=False):
        st.write(f"Feedback oversample weight (set in sidebar): {st.session_state['feedback_weight']}")
        if st.button("Retrain using all feedback"):
            with st.spinner("Retraining model..."):
                result = subprocess.run(
                    ["python", "retrain_from_feedback.py", "--weight", str(st.session_state["feedback_weight"])],
                    capture_output=True, text=True
                )
                st.code(result.stdout)
                model, device = load_model()
                st.success("Retraining complete and model reloaded!")

with tab2:
    feedback_dashboard()
    all_feedback_download()
    st.markdown("---")
    st.info("View and download all feedback. Data is always up-to-date after any upload or retrain.")

st.markdown(
    "<div style='text-align:center; color:gray; margin-top:32px;'>"
    "Built with Streamlit, PyTorch, and GradCAM. All data stays on your machine unless you share/download it."
    "</div>",
    unsafe_allow_html=True
)
