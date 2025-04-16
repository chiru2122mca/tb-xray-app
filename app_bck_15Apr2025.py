import streamlit as st
import torch
import torchvision.models as models
#from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import torchvision.transforms as transforms
import io
import tensorflow as tf
import numpy as np
import cv2
import tempfile
from PIL import Image
import base64
import gdown
import torch
from torchvision import models
from lime import lime_image
from skimage.segmentation import mark_boundaries

# === Set Streamlit Page Config ===
st.set_page_config(page_title="TB Detection from Chest X-rays", layout="wide")
st.title("🩺 TB Detection from Chest X-rays using Deep Learning")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Model parameters
NUM_CLASSES = 2
IMG_SIZE = 224
CLASS_NAMES = ['Normal Chest X-rays', 'TB Chest X-rays']  # Replace with your classes

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
])

# Model definitions
def load_vgg16_model(model_path):
    model = models.vgg16(pretrained=False)
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load the entire model
    # try:
    #     model = torch.load(model_path, map_location=device)
    #     print("VGG16 entire model loaded successfully from:", model_path)
    # except RuntimeError as e:
    #     print(f"Error loading model: {e}")
    #     raise

    return model.to(device).eval()

def load_resnet50_model(model_path):
    model = models.resnet50(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

#def load_efficientnet_b3_model(model_path):
   #model = EfficientNet.from_name('efficientnet-b3')
   #for param in model.parameters():
  #  param.requires_grad = False
    #model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
   # model.load_state_dict(torch.load(model_path, map_location=device))
   # return model.to(device).eval()

def load_densenet121_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Model loading function with dropdown selection
@st.cache_resource
def load_selected_model(model_name, model_paths):
    model_funcs = {
        'VGG16': load_vgg16_model,
        'ResNet50': load_resnet50_model,
       # 'EfficientNet-B3': load_efficientnet_b3_model,
        'DenseNet121': load_densenet121_model
    }
    if not os.path.exists(model_paths[model_name]):
        raise FileNotFoundError(f"Model file not found at: {model_paths[model_name]}")
    return model_funcs[model_name](model_paths[model_name])

# Predict function
def predict_image(model, image,model_name):
 if model_name == 'DenseNet121':  # Keras
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.image.resize(img_array, [IMG_SIZE, IMG_SIZE])
        img_array = np.expand_dims(img_array / 255.0, axis=0)
        predictions = model.predict(img_array)
        tb_prob = float(predictions[0][0])
        normal_prob = 1 - tb_prob 
        probs = [normal_prob, tb_prob]

        # Decide predicted class
        pred_index = 1 if tb_prob > 0.5 else 0
        pred_class = "TB Chest X-rays" if pred_index == 1 else "Normal Chest X-rays"
        
        heatmap = make_gradcam_keras(img_array, model, layer_name="conv5_block16_2_conv")
        overlay_img = overlay_heatmap(heatmap, image)
        
        return pred_class, probs,overlay_img
        
 else:  # PyTorch        
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = nn.Softmax(dim=1)(output)
        _, pred = torch.max(output, 1)
    return CLASS_NAMES[pred.item()], probs.cpu().numpy()[0]

# Image statistics
def get_image_stats(image):
    img_array = np.array(image)
    stats = {
        'Dimensions': f"{image.size[0]}x{image.size[1]}",
        'Mean Pixel Values (RGB)': np.mean(img_array, axis=(0, 1)).round(2).tolist(),
        'Size (KB)': f"{os.path.getsize(io.BytesIO(image.tobytes()).tell()) / 1024:.2f}"
    }
    return stats

def make_gradcam_keras(img_array, model, layer_name):
    """
    Generate Grad-CAM heatmap for Keras-based model (e.g., DenseNet121).
    
    Args:
        img_array (np.array): Preprocessed input image array of shape (1, H, W, 3).
        model (keras.Model): Trained Keras model.
        layer_name (str): Name of the last convolutional layer (e.g., "conv5_block16_2_conv").
        
    Returns:
        heatmap (np.array): Grad-CAM heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of class output w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling across width and height
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by its importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to 0–1
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + tf.keras.backend.epsilon())
    heatmap = heatmap.numpy()

    return heatmap
    
def overlay_heatmap(heatmap, original_image, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap onto the original image.
    
    Args:
        heatmap (np.array): Grad-CAM heatmap.
        original_image (PIL.Image): Original image before preprocessing.
        alpha (float): Transparency factor for heatmap.
        colormap (int): OpenCV colormap to use.
        
    Returns:
        overlay (PIL.Image): Blended image.
    """
    # Convert heatmap to 0–255 range
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, original_image.size)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)

    # Convert original PIL image to array
    original_array = np.array(original_image.convert("RGB"))

    # Blend heatmap with original image
    overlay = cv2.addWeighted(original_array, 1 - alpha, heatmap_color, alpha, 0)

    return Image.fromarray(overlay)

def make_gradcam_pytorch(model, image_tensor, target_layer):
    import numpy as np
    import cv2
    import torch
    from torch.nn import functional as F

    # Hook for gradients and activations
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # Forward pass
    model.eval()
    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()
    loss = output[0, pred_class]
    loss.backward()

    # Remove hooks
    handle_fw.remove()
    handle_bw.remove()

    # Compute Grad-CAM
    grads_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations_val[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    return cam

# Streamlit app
def main():
    # Model paths (update these to your local directory)
    model_paths = {
        'VGG16': 'best_vgg16_tuberculosis.pth',
        'ResNet50': 'best_resnet50_tuberculosis.pth',
        #'EfficientNet-B3': 'efficientnet_b3_brain_tumor.pth',
        'DenseNet121': 'tb_model.keras'
    }

    # Sidebar
    with st.sidebar:
        # st.title("Brain Tumor Classifier")
        # st.write("Choose a model and upload an MRI image to classify it into one of 4 categories.")
        # st.write(f"Classes: {', '.join(CLASS_NAMES)}")
        # st.write("Models available: VGG16, ResNet50, EfficientNet-B3, InceptionV3")

        # File uploader
        uploaded_file = st.file_uploader("Upload an X-Ray Image...", type=['jpg', 'jpeg', 'png'])
        # Model selection dropdown
        model_name = st.selectbox("Choose a Model", list(model_paths.keys()))

    # Main content
    st.header("TB Classification")
    # st.subheader("Select Model and Upload Image")

    if uploaded_file is not None:
    # Load and display image
    
        image = Image.open(uploaded_file).convert('RGB')
             
        # Load selected model
        try:
            model = load_selected_model(model_name, model_paths)
            st.success(f"{model_name} model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading {model_name}: {e}")
        
        # Predict
        pred_class, probs,overlay_img = predict_image(model, image,model_name)
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🩻 Original X-ray")
            st.image(image, caption="Original Image", width=300)

        with col2:
            st.markdown("### 🔥 Grad-CAM Heatmap")
            st.image(overlay_img, caption="Grad-CAM", width=300)
            st.markdown("""
        <div style='font-size: 14px; margin-top: -10px;'>
        <b>Color Legend:</b><br>
        <span style='color:red;'>Red/Yellow</span> = High importance (model focused here) <br>
        <span style='color:green;'>Green</span> = Moderate influence <br>
        <span style='color:blue;'>Blue</span> = Low importance (background / less relevant)
        </div>
        """, unsafe_allow_html=True)

        # Display prediction
        st.write("### Prediction Result")
        st.success(f"Predicted Class: **{pred_class}** (using {model_name})")
        st.write("Class Probabilities:")
        st.subheader("Confidence Bar")
        for cls, prob in zip(CLASS_NAMES, probs):
            st.write(f"{cls}: {prob:.4f}")
        st.progress(int(prob * 100))
        # Display image statistics
        st.write("### Image Statistics")
        stats = get_image_stats(image)
        for key, value in stats.items():
            st.write(f"{key}: {value}")

if __name__ == "__main__":
    main()
    
 # Footer
st.markdown("""
<div class="footer">
    🚀 Capstone Project | <b>BITS Pilani WILP</b> Group 17 | <span style="color:#ffd700;">2025</span> © <b>TB Detection Project</b>
</div>
""", unsafe_allow_html=True)
