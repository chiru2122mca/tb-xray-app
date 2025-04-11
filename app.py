import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from PIL import Image
import base64
import gdown
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from lime import lime_image
from skimage.segmentation import mark_boundaries

# === Set Streamlit Page Config ===
st.set_page_config(page_title="TB Detection from Chest X-rays", layout="wide")
st.title("🩺 TB Detection from Chest X-rays using Deep Learning")

# === Custom theme styles ===
# Custom theme styles
st.markdown("""
<style>
/* Entire background + font */
.stApp {
        background-image:;
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(255, 255, 255, 0.85);  /* Light white overlay */
        z-index: -1;
    }
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
    color: #1e3d59;
}

/* Fix dropdown text visibility and make it colorful */
div[data-baseweb="select"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 16px;
    font-weight: 500;
    color: #1e1e1e;
}

div[data-baseweb="select"] > div {
    width: 320px !important;
    background-color: #e6f0ff !important;
    border: 1px solid #80bfff !important;
    border-radius: 10px !important;
    color: #1e1e1e !important;
}

div[data-baseweb="select"] input {
    color: #1e1e1e !important;
    font-weight: 600;
    background-color: transparent;
}

div[data-baseweb="select"] [role="option"] {
    color: #000 !important;
    background-color: #fff !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background-color: #edf6ff;
    padding: 15px;
    border: 2px dashed #b3d7ff;
    border-radius: 12px;
}

/* Footer */
footer {
    font-size: 13px;
    color: #999;
}
</style>
""", unsafe_allow_html=True)

# === Model Options ===
model_choice = st.selectbox("Choose a Model:", [
    "DenseNet121",
    "ResNet50",
    "VGG16"
   # "EfficientNetB0"
], key="model_selector")

image_size = (224, 224)

# === Load Models with Caching ===
@st.cache_resource
def load_model(name):
    if name == "DenseNet121":
        return tf.keras.models.load_model("tb_model.keras")
    #elif name == "EfficientNetB0":
     #   return tf.keras.models.load_model("my_model0 (1)_l.keras")
    elif name == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load("best_resnet50_tuberculosis.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    elif name == "VGG16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
        model.load_state_dict(torch.load("best_vgg16_tuberculosis.pth", map_location=torch.device('cpu')))
        model.eval()
        return model

# === Grad-CAM for Keras ===
def make_gradcam_keras(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model([
        model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), predictions.numpy()

# === Grad-CAM for PyTorch ===
def make_gradcam_pytorch(model, img_tensor, final_conv, model_type="resnet"):
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    hook_handle_fwd = final_conv.register_forward_hook(forward_hook)
    hook_handle_bwd = final_conv.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(img_tensor)

    prob = torch.nn.functional.softmax(output, dim=1)[0][1].item()
    loss = output[0][1]

    loss.backward()
    grad = gradients[0][0].detach().numpy()
    fmap = features[0][0].detach().numpy()

    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, image_size)
    cam -= np.min(cam)
    cam /= np.max(cam)

    hook_handle_fwd.remove()
    hook_handle_bwd.remove()
    return cam, prob

# === Overlay Heatmap ===
def overlay_heatmap(heatmap, original_img, alpha=0.5):
    original_array = np.array(original_img)
    heatmap = cv2.resize(heatmap, (original_array.shape[1], original_array.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    return overlay

# === Image Upload ===
uploaded_file = st.file_uploader("Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize(image_size)
    model = load_model(model_choice)
    pred_class = ""
    pred_prob = 0.0

    if model_choice == "DenseNet121":
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array / 255.0, axis=0)
        heatmap, prediction = make_gradcam_keras(img_array, model, "conv5_block16_2_conv")
        overlay_img = overlay_heatmap(heatmap, img)
        pred_class = "TB Detected" if prediction[0][0] > 0.5 else "Normal"
        pred_prob = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    else:
        model_type = "resnet" if model_choice == "ResNet50" else "vgg"
        preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        img_tensor = preprocess(img).unsqueeze(0)
        final_conv = model.layer4[-1].conv3 if model_type == "resnet" else model.features[-1]
        heatmap, prob = make_gradcam_pytorch(model, img_tensor, final_conv, model_type)
        overlay_img = overlay_heatmap(heatmap, img)
        pred_class = "TB Detected" if prob > 0.5 else "Normal"
        pred_prob = prob if prob > 0.5 else 1 - prob

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded X-ray")
        st.image(img, caption="Original Image", width=250)
    with col2:
        st.subheader("Model Explanation (Grad-CAM)")
        st.image(overlay_img, caption=f"{pred_class} ({pred_prob:.2f} confidence)", width=250)
        st.markdown("""
        <div style='font-size: 14px; margin-top: -10px;'>
        <b>Color Legend:</b><br>
        <span style='color:red;'>Red/Yellow</span> = High importance (model focused here) <br>
        <span style='color:green;'>Green</span> = Moderate influence <br>
        <span style='color:blue;'>Blue</span> = Low importance (background / less relevant)
        </div>
        """, unsafe_allow_html=True)

    st.success(f"Prediction: {pred_class} with {pred_prob:.2f} confidence")
    st.subheader("Confidence Bar")
    st.progress(int(pred_prob * 100))

    if st.button("Download Explanation Image"):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, np.uint8(overlay_img * 255) if overlay_img.max() <= 1.0 else overlay_img)
        st.download_button(
            label="Download Image",
            data=open(temp_file.name, "rb").read(),
            file_name="explanation_output.jpg",
            mime="image/jpeg"
        )
else:
    st.info("Upload an image to start prediction.")

# Footer
st.markdown("""
<div class="footer">
    🚀 Capstone Project | <b>BITS Pilani WILP</b> Group 17 | <span style="color:#ffd700;">2025</span> © <b>TB Detection Project</b>
</div>
""", unsafe_allow_html=True)
