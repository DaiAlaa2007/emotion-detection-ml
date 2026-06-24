# Emotion Detection AI

A bilingual (Arabic & English) emotion detection system built using Machine Learning and FastAPI.

This project analyzes user text and predicts the underlying emotion using a Logistic Regression classifier trained on a custom dataset. It also includes a feedback mechanism that allows users to contribute new labeled examples for future model improvements.

---

## Overview

The goal of this project is to classify text into one of several emotional categories based on user input.

The system:

* Accepts Arabic and English text.
* Converts text into numerical features using TF-IDF.
* Predicts emotions using a Logistic Regression model.
* Returns confidence scores.
* Collects user feedback when the model is uncertain.
* Provides a simple web interface for interaction.

---

## Features

###  Emotion Classification

Detects the following emotions:

* 😊 Joy
* 😢 Sadness
* 😡 Anger
* 😨 Fear
* 😲 Surprise
* 😐 Neutral

### Bilingual Support

Supports both:

* Arabic
* English

###  Confidence-Based Prediction

The model returns a confidence score for every prediction.

If confidence is low, the system asks the user to select the correct emotion.

###  Feedback Collection

Users can submit the correct emotion when the model is uncertain.

The feedback is automatically stored in the dataset for future retraining.

###  REST API

Built using FastAPI for fast and lightweight deployment.

---

##  Technologies Used

* Python
* FastAPI
* Scikit-Learn
* Pandas
* TF-IDF Vectorization
* Logistic Regression
* HTML
* CSS
* JavaScript

---

##  Project Structure

Emotion-Detection-AI/

├── main.py

├── index.html

├── data.csv

├── requirements.txt

└── README.md

---

##  Machine Learning Pipeline

User Input

↓

Text Preprocessing

↓

TF-IDF Vectorization

↓

Logistic Regression Model

↓

Emotion Prediction

↓

Confidence Evaluation

↓

Feedback Collection

---

##  Installation

### Clone Repository

```bash
git clone https://github.com/your-username/emotion-detection-ai.git
cd emotion-detection-ai
```

### Install Dependencies

```bash
pip install fastapi
pip install uvicorn
pip install pandas
pip install scikit-learn
```

Or:

```bash
pip install -r requirements.txt
```

---

##  Running the Backend

```bash
uvicorn main:app --reload --port 8001
```

The API will run on:

```text
http://127.0.0.1:8001
```

---

## API Endpoints

### Predict Emotion

```http
POST /predict
```

Request:

```json
{
  "message": "حاسه مبسوطة النهارده"
}
```

Response:

```json
{
  "status": "ok",
  "emotion": "joy",
  "confidence": 0.91
}
```

---

### Submit Feedback

```http
POST /feedback
```

Request:

```json
{
  "text": "أنا متحمسة جدًا",
  "emotion": "joy"
}
```

Response:

```json
{
  "status": "saved"
}
```

---

##  Dataset

The model was trained on a custom bilingual dataset containing emotional expressions in both Arabic and English.

Dataset categories:

* Joy
* Sadness
* Anger
* Fear
* Surprise
* Neutral

---

## Learning Outcomes

This was my first Artificial Intelligence and Machine Learning project.

Through this project I learned:

* Text Classification
* Machine Learning Fundamentals
* TF-IDF Feature Extraction
* Logistic Regression
* Model Confidence Analysis
* FastAPI Development
* Frontend–Backend Integration
* Data Collection and Feedback Loops

---

##  Future Improvements

* Arabic NLP preprocessing
* Larger training dataset
* Automatic model retraining
* Deep Learning models (LSTM / Transformers)
* Hugging Face integration
* Emotion intensity detection
* Deployment on cloud platforms

---

##  Author

Dai Alaa


Interested in Artificial Intelligence, Software Development, Space Technology, and Research & Innovation.
