import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from PIL import Image
import base64

# === Set Streamlit Page Config ===
st.set_page_config(page_title="TB Detection from Chest X-rays", layout="wide")
st.title("🩺 TB Detection from Chest X-rays using Deep Learning")

# === Load the trained model ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tb_model.keras")

model = load_model()
last_conv_layer_name = "conv5_block16_2_conv"
image_size = (224, 224)

# === Utility functions ===
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    return np.expand_dims(img_array / 255.0, axis=0), np.array(img_resized)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
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

def overlay_heatmap(heatmap, original_img, alpha=0.5):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap_color, alpha, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    return overlay

def save_image(image, file_path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image_bgr)

# === Streamlit App UI ===


uploaded_file = st.file_uploader("Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_array, img_display = preprocess_image(uploaded_file)
    heatmap, prediction = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlay_img = overlay_heatmap(heatmap, img_display)

    pred_class = "TB Detected" if prediction[0][0] > 0.5 else "Normal"
    pred_prob = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded X-ray")
        st.image(img_display, caption="Original Image", width=250)

    with col2:
        st.subheader("Grad-CAM Heatmap")
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

    if st.button("Download Grad-CAM Image"):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        save_image(overlay_img, temp_file.name)
        st.download_button(
            label="Download Image",
            data=open(temp_file.name, "rb").read(),
            file_name="gradcam_output.jpg",
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
