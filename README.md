# osteoporosis

# 🦴 Osteoporosis Detection & AI Report Generator

This Streamlit-based AI tool analyzes spinal X-ray images to detect early signs of osteoporosis using computer vision and a deep learning classifier. It provides risk categorization, visual overlays (heatmaps, bounding boxes), and a professional report powered by Google Gemini AI.

---

## 📌 Features

- 📤 Upload grayscale X-ray images (`.jpg`, `.jpeg`, `.png`)
- 🧠 Automated detection of:
  - Cortical thinning
  - Trabecular degradation
  - Radiolucency
  - Compression fractures
  - Endplate irregularities
  - Geometric anomalies
- 📊 Risk categorization: **Low, Moderate, High, Critical**
- 🎯 Deep learning model prediction using PyTorch
- 🖼️ Visual overlays and Grad-CAM heatmaps
- 📝 AI-generated textual report via Gemini Pro (optional)

---

## 🧠 How It Works - project flow

1. Load pre-trained PyTorch model
2. Preprocess image (grayscale, resized to 224x224)
3. Extract 6 key diagnostic features
4. Predict risk using both feature-based and model-based methods
5. Visualize diagnostic overlays and Grad-CAM
6. Generate professional report (if Gemini API is enabled)

---

## 🧱 Project Structure

```
osteoporosis_project/
├── app.py                         # Streamlit app frontend
├── osteoporosis.py                # Core logic and model pipeline
├── best_osteoporosis_model.pth    # Trained PyTorch model
├── requirements.txt               # Python dependencies
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/osteoporosis.git
cd osteoporosis
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:
```bash
pip install streamlit torch torchvision opencv-python numpy matplotlib pillow scikit-image scikit-learn google-generativeai python-dotenv
```

---

## ▶️ Running the App Locally

```bash
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing the System

1. Run the app
2. Upload a valid grayscale X-ray image
3. Click **"Analyze & Generate Report"**
4. View:
   - Risk category
   - Feature-level scores
   - Diagnostic overlays and heatmaps
   - Full AI-generated medical summary

---

## 🧾 Sample Output

```
Overall Risk: High (Score: 0.78)
Primary Concerns: cortical_thinning, compression_fractures

Cortical Thinning: 0.88
Trabecular Degradation: 0.77
...

Grad-CAM heatmap generated ✅
Gemini Report Generated ✅
```

---

## 📈 Future Improvements

- Export PDF report with images and diagnosis
- Add user login / patient metadata
- Integrate DICOM support via `pydicom`
- Deploy on cloud using Hugging Face or Streamlit Cloud

---

## 🙌 Acknowledgments

- Google Gemini AI
- PyTorch for model inference
- OpenCV & scikit-image for image processing
- Streamlit for the UI
