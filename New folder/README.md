# 🩺 TB Detection from Chest X-rays using Deep Learning

This project is a Streamlit web application that uses a trained **DenseNet121** model to classify chest X-ray images as **TB (Tuberculosis) Positive** or **Normal**. It also provides **Grad-CAM heatmap visualizations** to explain the model’s decisions.

---

## 🚀 Features
- Upload chest X-ray images
- Predict TB vs. Normal using a deep learning model
- Visualize regions of attention with **Grad-CAM**
- Download the Grad-CAM result
- Elegant UI with banner, legend, and footer

---

## 📁 Folder Structure
```
tb-detection-app/
├── app.py                              # Streamlit app code
├── tb_model.keras                      # Trained DenseNet121 model
├── background.jpg                      # Background image (optional)
├── A_digital_graphic_displays_Grad-CAM_Heatmap_interp.png  # Legend image
├── requirements.txt                    # Dependencies
├── README.md                           # You're here
```

---

## 📦 Setup Instructions

### 🔧 Install Dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Run Locally
```bash
streamlit run app.py
```

### ☁️ Deploy Online (Streamlit Cloud)
1. Push this folder to a **public GitHub repo**
2. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **New App** and choose your repo
4. Set `app.py` as the entry point and deploy 🎉

---

## 🧠 Model Info
- Architecture: **DenseNet121**
- Framework: **TensorFlow / Keras**
- Trained to detect: **TB lesions, infiltrates, and abnormalities** in chest X-rays

---

## ✨ Credits
Capstone Project | **BITS Pilani WILP Group 17** | 2025 © TB Detection Project

---

## 📬 Feedback / Contact
Feel free to [create an issue](https://github.com/) or connect with us on LinkedIn.

---

> Made with ❤️ for better healthcare outcomes using AI
