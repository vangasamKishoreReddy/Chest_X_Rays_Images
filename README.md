# Chest_X_Rays_Images


# 🩻 Chest X-Ray Pneumonia Classification Web App

An interactive **Streamlit web application** for classifying chest X-ray images as **Normal** or **Pneumonia** using a **ResNet-50 deep learning model**, with **Grad-CAM visualizations**, **feature descriptors (ORB & SIFT)**, and performance metrics.

## 🚀 Demo Features

- ✅ Upload X-ray images and receive real-time predictions
- 📊 Visualize Grad-CAM heatmaps highlighting critical lung regions
- 🧠 Includes handcrafted features (ORB, SIFT) for interpretability
- 📈 Live performance evaluation: confusion matrix, classification report
- 🖼️ Explore dataset samples with class breakdowns
- 📂 Integrated test, train, validation data loader
- 🧠 Built with PyTorch, OpenCV, Streamlit, and torchvision

---

## 📦 Project Structure

├── main.py # Main Streamlit app

├── models/ # Model definition and weights

│ └── best_model.pth

├── data/

│ ├── train/val/test/ # Processed .npy, .pkl and descriptors

├── utils/ # Feature visualization (ORB, SIFT)

├── outputs/ # GradCAM visualization image

├── requirements.txt # Python dependencies

└── README.md # You are here!

## 📋 Requirements

  Python 3.8+
  
  torch
  
  torchvision
  
  numpy
  
  matplotlib
  
  scikit-learn
  
  streamlit
  
  opencv-python
  
  seaborn
  
  pandas
  
  Pillow

## Deployement
streamlit run main.py

## 📊 Results Snapshot

  Test Accuracy: ~90%
  
  AUC Score: 0.92+
  
  F1-Score: 0.89
  
  Grad-CAM explains why the model chose pneumonia/normal
