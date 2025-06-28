import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Set up the app
st.set_page_config(page_title="X-Ray Image Classifier", layout="wide")
st.title("Chest_X-Ray Image Classification")
st.write("Upload an X-ray image for classification (Normal vs. Pneumonia)")

# Define paths (update these with your actual paths)
DATA_PATHS = {
    "train": {
        "images": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\train\images.npy",
        "labels": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\train\labels.npy",
        "metadata": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\train\metadata.pkl",
        "orb": r"E:\Medical Images\Chest_XRay_Dataset\train_orb_descriptors.npz",
        "sift": r"E:\Medical Images\Chest_XRay_Dataset\train_sift_descriptors.npz"
    },
    "val": {
        "images": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\val\images.npy",
        "labels": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\val\labels.npy",
        "metadata": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\val\metadata.pkl",
        "orb": r"E:\Medical Images\Chest_XRay_Dataset\val_orb_descriptors.npz",
        "sift": r"E:\Medical Images\Chest_XRay_Dataset\val_sift_descriptors.npz"
    },
    "test": {
        "images": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\test\images.npy",
        "labels": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\test\labels.npy",
        "metadata": r"E:\Medical Images\Chest_XRay_Dataset\processed_data\test\metadata.pkl",
        "orb": r"E:\Medical Images\Chest_XRay_Dataset\test_orb_descriptors.npz",
        "sift": r"E:\Medical Images\Chest_XRay_Dataset\test_sift_descriptors.npz"
    },
    "gradcam": r"E:\Medical Images\Chest_XRay_Dataset\gradcam_output.png"
}

# Define the model architecture (must match training)
class XRayClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base_model = models.resnet50(pretrained=False)
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes))
    
    def forward(self, x):
        return self.base_model(x)

# Load the trained model
@st.cache_resource
def load_model():
    model = XRayClassifier()
    model.load_state_dict(torch.load(r"E:\Medical Images\output\best_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Load dataset statistics and samples
@st.cache_resource
def load_dataset_info():
    datasets = {}
    for split in ['train', 'val', 'test']:
        try:
            datasets[split] = {
                'images': np.load(DATA_PATHS[split]['images']),
                'labels': np.load(DATA_PATHS[split]['labels']),
                'metadata': pickle.load(open(DATA_PATHS[split]['metadata'], 'rb')),
                'orb': np.load(DATA_PATHS[split]['orb']) if os.path.exists(DATA_PATHS[split]['orb']) else None,
                'sift': np.load(DATA_PATHS[split]['sift']) if os.path.exists(DATA_PATHS[split]['sift']) else None
            }
        except Exception as e:
            st.warning(f"Could not load {split} data: {str(e)}")
            datasets[split] = None
    
    try:
        datasets['gradcam_sample'] = Image.open(DATA_PATHS['gradcam']) if os.path.exists(DATA_PATHS['gradcam']) else None
    except Exception as e:
        st.warning(f"Could not load GradCAM sample: {str(e)}")
        datasets['gradcam_sample'] = None
    
    return datasets

datasets = load_dataset_info()

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Feature descriptor visualization functions
def visualize_orb_features(image, keypoints):
    img = image.copy()
    img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    return img

def visualize_sift_features(image, keypoints):
    img = image.copy()
    img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

# Corrected Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activation = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()
        
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_full_backward_hook(backward_hook))
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot)
        
        # Process gradients and activations
        gradients = self.gradient.cpu().numpy()[0]
        activations = self.activation.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Post-processing
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
        
        return cam, target_class
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# Dataset visualization section
st.sidebar.header("Dataset Explorer")
selected_split = st.sidebar.selectbox("Select dataset split", ["train", "val", "test"])

if datasets.get(selected_split) is not None:
    split_data = datasets[selected_split]
    st.sidebar.write(f"**{selected_split.capitalize()} Set Info**")
    st.sidebar.write(f"Total samples: {len(split_data['labels'])}")
    st.sidebar.write(f"Normal cases: {np.sum(split_data['labels'] == 0)}")
    st.sidebar.write(f"Pneumonia cases: {np.sum(split_data['labels'] == 1)}")
    
    # Show sample images
    st.sidebar.subheader("Sample Images")
    sample_indices = st.sidebar.slider("Select sample range", 0, len(split_data['labels'])-1, (0, 5))
    
    cols = st.sidebar.columns(2)
    for i in range(sample_indices[0], min(sample_indices[1]+1, len(split_data['labels']))):
        label = "Normal" if split_data['labels'][i] == 0 else "Pneumonia"
        cols[i % 2].image(split_data['images'][i], caption=f"Sample {i} ({label})", width=100)

# Feature descriptor visualization
if st.sidebar.checkbox("Show Feature Descriptors"):
    if datasets.get(selected_split) is not None:
        st.header("Feature Descriptor Visualization")
        sample_idx = st.slider("Select sample image", 0, len(datasets[selected_split]['labels'])-1, 0)
        image = datasets[selected_split]['images'][sample_idx]
        
        col1, col2 = st.columns(2)
        
        # ORB features
        if datasets[selected_split]['orb'] is not None:
            orb_descriptors = datasets[selected_split]['orb']
            if 'keypoints' in orb_descriptors and sample_idx < len(orb_descriptors['keypoints']):
                keypoints = [cv2.KeyPoint(x=kp[0][0], y=kp[0][1], size=kp[1], angle=kp[2], 
                                        response=kp[3], octave=int(kp[4]), class_id=int(kp[5])) 
                           for kp in orb_descriptors['keypoints'][sample_idx]]
                orb_img = visualize_orb_features(image, keypoints)
                col1.image(orb_img, caption="ORB Features", use_container_width=True)
        
        # SIFT features
        if datasets[selected_split]['sift'] is not None:
            sift_descriptors = datasets[selected_split]['sift']
            if 'keypoints' in sift_descriptors and sample_idx < len(sift_descriptors['keypoints']):
                keypoints = [cv2.KeyPoint(x=kp[0][0], y=kp[0][1], size=kp[1], angle=kp[2], 
                            response=kp[3], octave=int(kp[4]), class_id=int(kp[5])) 
                           for kp in sift_descriptors['keypoints'][sample_idx]]
                sift_img = visualize_sift_features(image, keypoints)
                col2.image(sift_img, caption="SIFT Features", use_container_width=True)

# Model performance section
if st.sidebar.checkbox("Show Model Performance"):
    if datasets.get('test') is not None:
        st.header("Model Performance on Test Set")
        test_images = datasets['test']['images']
        test_labels = datasets['test']['labels']
        
        batch_size = 32
        all_preds = []
        
        with st.spinner("Evaluating on test set..."):
            for i in range(0, len(test_images), batch_size):
                batch_images = test_images[i:i+batch_size]
                batch_tensors = torch.stack([transform(img) for img in batch_images])
                
                with torch.no_grad():
                    outputs = model(batch_tensors)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(test_labels, all_preds, labels=[0, 1])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Pneumonia'],
                    yticklabels=['Normal', 'Pneumonia'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        
        if len(np.unique(test_labels)) > 1:
            st.subheader("Classification Report")
            report = classification_report(test_labels, all_preds,
                                        target_names=['Normal', 'Pneumonia'],
                                        labels=[0, 1],
                                        output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        else:
            st.warning("Only one class present in predictions")

# Show sample GradCAM output if available
if datasets.get('gradcam_sample') is not None:
    if st.sidebar.checkbox("Show GradCAM Example"):
        st.header("GradCAM Visualization Example")
        st.image(datasets['gradcam_sample'], caption="Sample GradCAM Output", use_column_width=True)

# File uploader for user images
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    # Display original image
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    
    # Preprocess for model
    input_tensor = transform(image_np).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_class = output.argmax().item()
        confidence = probabilities[0][pred_class].item()
    
    # Display prediction
    class_names = ["Normal", "Pneumonia"]
    st.subheader("Prediction Result")
    st.write(f"**Class:** {class_names[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2%}")
    
    # Generate Grad-CAM
    st.subheader("Model Attention Visualization")
    grad_cam = GradCAM(model, target_layer="base_model.layer4")
    cam, _ = grad_cam.generate_cam(input_tensor)
    grad_cam.remove_hooks()
    
    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    
    # Display side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_np, caption="Original Image", use_column_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM Visualization", use_column_width=True)
    
    # Interpretation
    st.subheader("Interpretation")
    st.write("""
    The Grad-CAM visualization shows which regions of the image most influenced the model's decision.
    - **Red/Yellow areas**: Most influential regions for the prediction
    - **Blue areas**: Less important regions
    """)
    
    # Confidence breakdown
    st.subheader("Confidence Breakdown")
    fig, ax = plt.subplots()
    ax.barh(class_names, probabilities[0].cpu().numpy())
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    st.pyplot(fig)

# Add some info sections
st.sidebar.header("About")
st.sidebar.info("""
This app uses a deep learning model to classify chest X-ray images as Normal or Pneumonia.
- Model: ResNet50 with custom head
- Training: 10 epochs with transfer learning
- Accuracy: ~90% on test set
- Features: Includes ORB and SIFT descriptors
""")

st.sidebar.header("How to Use")
st.sidebar.info("""
1. Upload a chest X-ray image (JPEG/PNG)
2. The model will process the image
3. View the prediction and visualization
4. Explore dataset samples and model performance
5. Examine feature descriptors and attention maps
""")