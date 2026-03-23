🥔 Potato Disease Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

---
Objective

To Helps farmers reduce crop loss automatically detect potato plant diseases from leaf images using a deep learning model and deploy it as a web application.

Step-1 📖 Description

This project is a deep learning-based web application that detects potato leaf diseases using image classification.
It uses **MobileNetV2 (Transfer Learning)** with TensorFlow and provides real-time predictions through a **Streamlit web interface**.

---

## 🚀 Features

* 🔍 Detects potato diseases from leaf images
* ⚡ Real-time prediction using Streamlit
* 📊 High accuracy (~98–99%)
* 🧠 Built using transfer learning (MobileNetV2)
* 🖼️ Simple and user-friendly interface

---

## 🧠 Model Details

* **Model:** MobileNetV2
* **Framework:** TensorFlow / Keras
* **Accuracy:** ~98–99%
* **Classes:**

  * Early Blight
  * Late Blight
  * Healthy

---

Step-2 🗂️ Project Structure

```
potato-disease-prediction/
│── app.py
│── train.py
│── potato_disease_model.h5
│── requirements.txt
```

---
Step-3 📂 Dataset Used

PlantVillage Dataset (public & trusted)

Source: Kaggle-https://www.kaggle.com/datasets/emmarex/plantdisease
Classes:
Potato___Early_blight
Potato___Late_blight
Potato___healthy

Dataset Size (Approx.)
~2,100 images
RGB images
Good quality, labeled

---
Step-4 ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/Anushka-0990/potato-disease-detection.git
```

2. Navigate to the project folder:

```
cd PlantVillage
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. streamlit run app.py

---

Step-5 ▶️ Run the App

```
  Local URL: http://localhost:8501
  Network URL: http://172.20.10.4:8501
```

---

Step-6 📸 App Preview
<img width="1867" height="1009" alt="Screenshot 2026-03-24 004619" src="https://github.com/user-attachments/assets/5cd94df1-8b56-40d4-ab26-4be9aab140c6" />
<img width="1872" height="1009" alt="Screenshot 2026-03-24 004656" src="https://github.com/user-attachments/assets/bd77c344-6a5a-4fca-91fe-19075026bea8" />
<img width="1864" height="1001" alt="Screenshot 2026-03-24 004741" src="https://github.com/user-attachments/assets/4324d238-07f4-4b36-8a8c-434884130564" />

---


https://github.com/user-attachments/assets/1ec664bb-4d8b-4bd1-86c4-f1854ad7c981



Step-7 🌐 Live Demo

*https://leafmatters.streamlit.app/*

    
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Streamlit
* NumPy
* Matplotlib

---

## 🤝 Contributing

Contributions are welcome!
Feel free to open an issue or submit a pull request.

---


