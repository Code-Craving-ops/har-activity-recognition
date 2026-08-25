# Human Activity Recognition using Machine Learning

## 📌 Project Overview

Human Activity Recognition (HAR) is a Machine Learning project that predicts a person's physical activity using sensor-based data.

The system analyzes movement-related sensor readings and uses a trained Machine Learning model to classify activities such as **walking, sitting, standing, running, and other human movements**.

This project demonstrates how Machine Learning can be applied to sensor and time-series data for activity monitoring and smart-device applications.

---

## 🎯 Objectives

* Build a Machine Learning model for human activity recognition.
* Process and prepare sensor-based activity data.
* Train and evaluate a classification model.
* Predict human activities from new sensor readings.
* Provide an easy way to use the trained model through a Python application.

---

## 🧠 How the System Works

```text
Sensor Data
     ↓
Data Preprocessing
     ↓
Feature Scaling
     ↓
Trained ML Model
     ↓
Activity Prediction
     ↓
Human Activity
```

The input sensor data is first preprocessed and scaled using the same transformation used during model training. The trained model then predicts the corresponding activity.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas** – Data manipulation
* **NumPy** – Numerical computations
* **Scikit-learn** – Machine Learning
* **Jupyter Notebook** – Model development and analysis
* **Pickle** – Saving trained ML components
* **Streamlit/Python Application** – Model interaction

---

## 📂 Project Structure

```text
har-activity-recognition/
│
├── app.py
├── har_model.pkl
├── scaler.pkl
├── label_encoder.pkl
├── requirements.txt
├── HAR.ipynb
└── README.md
```

### File Description

| File                | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `app.py`            | Application for making activity predictions           |
| `har_model.pkl`     | Trained Machine Learning model                        |
| `scaler.pkl`        | Feature scaling object                                |
| `label_encoder.pkl` | Encodes and decodes activity labels                   |
| `HAR.ipynb`         | Data analysis, preprocessing, training and evaluation |
| `requirements.txt`  | Required Python libraries                             |
| `README.md`         | Project documentation                                 |

## 📊 Machine Learning Workflow

The project follows a standard Machine Learning pipeline:

1. **Data Collection**
2. **Data Preprocessing**
3. **Exploratory Data Analysis**
4. **Feature Preparation**
5. **Feature Scaling**
6. **Model Training**
7. **Model Evaluation**
8. **Activity Prediction**

---

## 🔍 Applications

Human Activity Recognition can be useful in several real-world applications:

* 📱 Smartphone activity tracking
* ⌚ Smartwatch and wearable devices
* 🏃 Fitness and health monitoring
* 🏥 Healthcare applications
* 👴 Elderly activity monitoring
* 🤖 IoT and smart-device systems
* 🚨 Fall and abnormal activity detection

---

## 🚀 Future Improvements

The project can be further enhanced by adding:

* Real-time sensor data collection
* Fall detection
* Activity history and visualization
* Real-time activity dashboard
* Multiple Machine Learning models for comparison
* Deep Learning models such as LSTM
* Integration with smartphone or smartwatch sensors
* Personalized activity monitoring

---

## 📈 Future Scope

The system can be extended into a complete real-time activity monitoring platform. Sensor data from smartphones or wearable devices can be continuously processed to identify activities and detect unusual movements.

Adding **fall detection and real-time alerts** could make the project more useful for healthcare and elderly monitoring applications.

---

## 👨‍💻 Author

**Anuj Kumar**

This project is developed for learning and demonstrating the application of Machine Learning to Human Activity Recognition.

---

## 📄 License

This project is intended for educational and learning purposes. Please refer to the repository's license for usage and redistribution terms.
