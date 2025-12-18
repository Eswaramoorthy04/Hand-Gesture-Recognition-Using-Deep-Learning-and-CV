# Hand Gesture Recognition Using Deep Learning

## 📌 Overview
This project focuses on developing a **hand gesture recognition model** that can accurately identify and classify different hand gestures from image or video data.  
The system enables **intuitive human–computer interaction (HCI)** by allowing gesture-based control using deep learning and computer vision techniques.

A **Convolutional Neural Network (CNN)** is trained on image data and deployed for **real-time gesture recognition** using a webcam.

---

## 🎯 Objective
- To design a deep learning model capable of recognizing hand gestures  
- To classify gestures from image and video input  
- To enable real-time gesture-based interaction using a webcam  

---

## 🧠 Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV (cv2)  

---

## 📊 Dataset
- Image-based gesture dataset (28×28 grayscale format)  
- Data is preprocessed and normalized for CNN training  
- Classes represent different hand gestures  

*(Note: The model structure is compatible with MNIST-style image datasets and can be extended to custom gesture datasets.)*

---

## ⚙️ Data Preprocessing
- Image resizing to 28 × 28 pixels  
- Conversion to grayscale  
- Normalization of pixel values  
- One-hot encoding of labels  

---

## 🏗️ Model Architecture
The CNN model includes:
- Convolutional layers with ReLU activation  
- Max pooling layers for feature extraction  
- Fully connected dense layers  
- Softmax output layer for multi-class gesture classification  

---

## 🧪 Model Training
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Batch Size: 128  
- Epochs: 10  
- Validation performed using test data  

---

## 📈 Model Evaluation
- Evaluated using test dataset  
- Metrics used:
  - Accuracy  
  - Loss  

The trained model achieves reliable performance in gesture classification.

---

## 🎥 Real-Time Gesture Recognition
- Webcam captures live video frames  
- Frames are preprocessed and passed to the trained CNN  
- The predicted gesture is displayed in real time  
- Press **'q'** to exit the application  

---

## ✅ Results
- Successfully classifies hand gestures from image input  
- Supports real-time prediction using webcam feed  
- Demonstrates effective application of CNNs for HCI systems  

---

## 🚀 Future Enhancements
- Train on a dedicated hand gesture dataset  
- Improve accuracy using deeper CNN architectures  
- Add gesture tracking and bounding box detection  
- Deploy as a web or mobile application  

---
