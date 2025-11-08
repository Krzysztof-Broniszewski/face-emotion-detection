# Real-time Facial Emotion Recognition 🎭  
**OpenCV + MediaPipe + CLIP (PyTorch)**

Real-time emotion recognition using **OpenAI CLIP** and **MediaPipe FaceMesh**, running directly on a live webcam feed.  
No dataset training required — emotions are recognized in a *zero-shot* fashion using CLIP embeddings.

---

## 🎬 Live Demo

![Live Demo](./Emotion_detector.gif)

*(Recorded with [ScreenToGif](https://www.screentogif.com/), showing live OpenCV feed with zero-shot CLIP emotion labels)*

---

## 🧠 Model Overview

This project is based on the **CLIP (Contrastive Language–Image Pretraining)** model  
originally developed by **OpenAI**, integrated via the **Hugging Face Transformers** library.

- **Base model:** `openai/clip-vit-large-patch14`
- **Architecture:** Vision Transformer (ViT-L/14)
- **Embedding dimension:** 1024
- **Zero-shot capability:** No fine-tuning required – it matches face images to emotion text prompts
- **Framework:** PyTorch + HuggingFace
- **Weights format:** `.safetensors` (1.2 GB)

### ⚙️ How it works
CLIP processes both:
- the **face image** (as a visual embedding), and  
- the **emotion descriptions** (as textual embeddings, e.g. “happy face”, “sad face”),  
and computes the cosine similarity between them.  
The emotion with the highest similarity score is chosen as the prediction.

## 🔍 Features
- Real-time face detection using **MediaPipe** (468-point mesh not required here)
- Automatic extraction of face ROI (48×48 grayscale)
- Emotion recognition with **CLIP zero-shot model**
- Display of emotion probabilities in percentage
- Support for **Iriun Webcam** and standard webcams
- Multilingual labels (English / Polish)

---

Although the model was **not fine-tuned** on FER2013 or AffectNet, it achieves about:

| Dataset | Accuracy (zero-shot, reported in research) |
|----------|--------------------------------------------|
| Custom subset (FER-like) | ~52–58 % |
| After light fine-tuning  | up to ~80 % |

This explains why results may vary — the model generalizes well but sometimes misclassifies subtle or occluded expressions.

## Weights
Large weights are NOT committed to the repo.

- CLIP vision weights (~1.2 GB) are auto-downloaded from Hugging Face on first run.
- Optional prefetch:
  ```bash
  python scripts/prefetch_weights.py

## 🧠 Recognized emotions
| English       | Polish        |
|----------------|----------------|
| Happy face     | Radość         |
| Sad face       | Smutek         |
| Angry face     | Złość          |
| Surprised face | Zaskoczenie    |
| Disgusted face | Wstręt         |
| Fearful face   | Strach         |
| Neutral face   | Neutralny      |

---

## 🧩 Tech Stack
- **Python 3.11**
- **PyTorch**
- **Transformers / HuggingFace CLIP**
- **OpenCV**
- **MediaPipe**
- **NumPy / Pillow**
- **Jupyter Notebook**

---

## 📸 Example Output
Below is a sample frame from the live video feed showing face detection and real-time emotion probabilities:

![Example Screenshot](./face_mesh_468_points.png)

*(Located in the project folder — `face_mesh_468_points.png`)*

![Example Screenshot](./face_demo.png)

*(Located in the project folder — `face_demo.png`)*

*(Replace with your actual screenshot — e.g., saved as `assets/screenshot_emotion_detection.jpg` in your repo.)*

---


## 🧩 Future Improvements
- Fine-tune the CLIP vision encoder on **FER2013** or **AffectNet**  
- Add temporal smoothing (EMA or rolling average) for stable predictions  
- Implement Streamlit or Gradio demo with bilingual UI  
- Evaluate accuracy quantitatively on FER2013 test split  

## 🚀 Run Locally

### 1️⃣ Clone and set up the environment
```bash
git clone https://github.com/yourusername/face-emotion-detection.git
cd face-emotion-detection
conda activate ml-pytorch-face
