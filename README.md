# DASHCAM-ANOMALY-CLASSIFIER
Deep learning-based dashcam anomaly classifier using TensorFlow, scikit-learn, CNNs, and OpenCV. Built on Google Colab with T4 GPU to detect traffic violations, collisions, and unusual road events from video footage.
## 🚦 Project Overview

This project is a deep learning-based **Dashcam Anomaly Classifier** that uses video frame data to detect accidents or unusual road events. It leverages TensorFlow, CNNs, scikit-learn, and OpenCV. Built and trained on Google Colab with T4 GPU support.

The classifier predicts whether a frame shows an accident (`accident`) or normal traffic (`non_accident`).
# 🚦 Dashcam Anomaly Classifier

A deep learning-based binary classifier that detects accidents or unusual road anomalies in dashcam footage using frame-level image data. Built with TensorFlow, trained on accident/non-accident image datasets, and evaluated using standard metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

---

## 📁 Dataset

The dataset was structured into:

It contained a total of:
- **52,499** training images
- **11,250** validation images
- **11,251** test images

---

## 🧠 Model Architecture

- Convolutional Neural Network (CNN)
- L2 regularization (to reduce overfitting)
- Dropout layer (0.5 rate)
- Binary classification (sigmoid output)

---

## 📊 Training & Evaluation

- Trained over **10 epochs** (initial 4 + resumed 6)
- Early stopping and model checkpointing applied
- Data augmentation applied to training data

#### 📈 Performance Metrics:
- **Accuracy**: 78%
- **Precision**: 0.78
- **Recall**: 0.20 _(accidents often underdetected)_
- **F1 Score**: 0.32 _(accident class)_

> 📌 See `evaluation_metrics.txt` for detailed scores.

---

## 📉 Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

- High accuracy for **non-accident detection**
- Low recall on **accident frames** due to class imbalance and visual similarity

---

## 🛠️ How to Run

1. Open the notebook `Dashcam_Anomaly_Classifier.ipynb` in Google Colab.
2. Upload the dataset or use API to fetch.
3. Run all cells to:
   - Split dataset
   - Train model
   - Evaluate on test set
   - Generate plots and reports

---

## 💾 Files Included

| File                    | Description                                  |
|-------------------------|----------------------------------------------|
| `Dashcam_Anomaly_Classifier.ipynb` | Full Colab training notebook             |
| `best_model.h5`         | Best model weights (saved checkpoint)        |
| `confusion_matrix.png`  | Visual confusion matrix of test predictions  |
| `evaluation_metrics.txt`| Final accuracy, precision, recall, F1        |
| `final_report.txt`      | Text summary of model performance            |

---

## ✍️ Author

- **Saad Bin Masud** — [@IEncryptSaad](https://github.com/IEncryptSaad)

---

## 📌 Note

This project is focused on detecting anomalies in dashcam video frames, not real-time video analysis. For real-time detection, a video frame extractor + real-time inference loop is required.

