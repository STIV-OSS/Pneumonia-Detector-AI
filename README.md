# Pneumonia Detector AI – Technical Documentation

## 1. Overview

**Pneumonia Detector AI** is an explainable, user-feedback-driven deep learning web app for automated chest X-ray analysis.
It empowers clinicians and researchers with transparent AI predictions, on-the-fly human feedback and retraining, batch processing, and secure on-prem deployment.

---

## 2. System Architecture

* **Frontend:** Streamlit web application for interaction and visualization
* **Backend/ML:** PyTorch-based EfficientNet-B0 convolutional neural network
* **Explainability:** GradCAM attention heatmaps highlight the basis for every model decision
* **Data:** Chest X-ray images (user-uploaded or batch)
* **Feedback Loop:** All user corrections are stored for retraining
* **Retraining:** Triggered via UI, incorporates user feedback to continuously refine the model
* **Deployment:** Fully Dockerized for easy, reproducible deployment (CPU/GPU/server/workstation)

---

## 3. Key Features

* **AI Prediction:** Classifies each X-ray as “Normal” or “Pneumonia” and shows model confidence
* **Attention Map:** GradCAM overlays visually explain every AI decision
* **Image Quality Assurance:** Warns users of input images that are too small, too dark, or likely not valid chest X-rays
* **Batch Mode:** Accepts ZIP files for fast batch analysis and feedback
* **Feedback Loop:** Users can correct predictions and submit feedback to improve the model
* **Retraining:** Incorporates user feedback for continual model improvement, with adjustable feedback weight
* **Downloadable Reports:** Export feedback as CSV, PDF, JSON, or FHIR (for medical standards)
* **Audit Logging:** Tracks all retraining events, model hashes, and test metrics for compliance
* **Session Tracking:** Anonymous session IDs for privacy-respecting feedback logs
* **Health Endpoint:** Built-in health check for Docker/cloud deployments
* **Professional, Responsive UI:** Clean layout, accessible on desktop and mobile

---

## 4. User Flow

1. **Consent & Privacy:** Users see a disclaimer and must agree before using the app
2. **Upload:** Upload single/multiple X-rays or a ZIP for batch mode; sample image provided for demo/testing
3. **Prediction:** Model provides result, confidence, and an attention map per image
4. **Image Quality Check:** App warns if input is poor or invalid
5. **Feedback:** User can mark/correct each result and submit feedback
6. **Download:** Export session feedback in several formats
7. **Dashboard:** View/download all feedback history
8. **Retrain:** Update the model with new feedback; admin can adjust feedback weight for impact
9. **Repeat:** The model improves as it learns from more user-provided feedback

---

## 5. Technical Stack

| Layer          | Technology/Package                         |
| -------------- | ------------------------------------------ |
| UI/UX          | Streamlit                                  |
| Model          | PyTorch, EfficientNet-B0                   |
| Explainability | pytorch-grad-cam                           |
| Data Handling  | Pillow, Pandas, CSV, JSON, ReportLab (PDF) |
| Feedback       | CSV logging, session management            |
| Retrain        | Python script, CLI arguments               |
| Export         | CSV, PDF, JSON, FHIR (HL7)                 |
| Deployment     | Docker (Linux/Windows, CPU/GPU)            |

---

## 6. Data Description

* The model is trained and validated on a **diverse collection of anonymized, labeled chest X-ray images** representing both healthy individuals and patients diagnosed with pneumonia.
* Images have been curated and preprocessed for high-quality, robust AI learning. Class balance, labeling, and data partitioning follow accepted medical imaging research standards.

---

## 7. Screenshots

* Consent/Privacy Banner
<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/consent-banner.png" alt="Consent/Privacy Banner" width="480"/>
</p>
* Upload and Prediction UI
<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/upload-ui.png" alt="* Upload and Prediction UI" width="480"/>
</p>
* Attention Map Display
<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/attention-map.png" alt="* Attention Map Display" width="480"/>
</p>
* Image Quality Warning Example
<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/image-quality.png" alt="* Image Quality Warning Example" width="480"/>
</p>
* Feedback Submission
<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/feedback-submission.png" alt="* Feedback Submission" width="480"/>
</p>
* Session Feedback Export
<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/session-feedback.png" alt="* Session Feedback Export" width="480"/>
</p>
* Feedback Dashboard Tab
<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/feedback-dashboard.png" alt="* Feedback Dashboard Tab" width="480"/>
</p>
* Retrain Button & Logs
<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/retrain-status.png" alt="* Retrain Button & Logs" width="480"/>
</p>

<p align="center">
  <img src="https://github.com/twishapatel12/Pneumonia-Detector-AI/blob/main/assets/retrain-model.png" alt="* Retrain Button & Logs" width="480"/>
</p>

---

## 8. Model & Metrics

* **Model:** EfficientNet-B0 CNN, fine-tuned on curated chest X-ray images
* **Input:** 224x224 grayscale images, normalized
* **Augmentation:** Random rotations, flips, color jitter for robustness
* **Explainability Layer:** Final convolution layer (EfficientNet-B0)
* **Current Performance:**

  * Accuracy: 92%
  * AUC: 0.97
  * Pneumonia F1: 0.94
  * Normal F1: 0.89
* **Metrics are continuously updated and displayed in the UI after each retrain**

---

## 9. Security & Privacy

* All images and feedback remain **local** to the deployment environment (no cloud upload)
* **Anonymous session IDs** ensure user privacy
* **Audit logs** provide full traceability of all model retraining events
* **Health endpoint** allows robust monitoring when deployed in production

---

## 10. Deployment Instructions

1. **Build Docker image:**

   ```bash
   docker build -t pneumonia-ai-app .
   ```
2. **Run container:**

   ```bash
   docker run -p 8501:8501 pneumonia-ai-app
   ```
3. **Open browser to:**
   [http://localhost:8501](http://localhost:8501)

---

## 11. Limitations & Next Steps

* Not intended for use as a sole diagnostic tool—best for clinical research, education, or AI prototyping
* Next steps:

  * Expand to detect additional pathologies (multi-class support)
  * Integrate with hospital PACS/DICOM systems
  * Support for more advanced privacy/compliance requirements
  * Add role-based authentication for multi-user hospital pilots

---

## 12. Contact

* **Name:** *Twisha Patel*
* **Email:** *twishap534@gmail.com*
* **GitHub:** *https://github.com/twishapatel12*

---

## 13. Appendix

* Sample X-ray images, feedback CSV/PDF examples
* Retrain script usage example
* Audit log format
* Full Dockerfile
* Automated test stub
