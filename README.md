# 📧 Spam Email Detection System

A Machine Learning–based web application that classifies emails as **Spam** or **Not Spam** using **Natural Language Processing (NLP)** and **Naive Bayes**.  
The project is built with **Python**, **Scikit-learn**, and **Flask**, and deployed on **Render**.

---

## 🌐 Live Demo
🔗 **Live Application:**  
👉 https://spam-email-detection.onrender.com  

> ⚠️ Note: The application is deployed on Render free tier, so the first request may take a few seconds due to cold start.

---

## 🚀 Features
- Detects spam emails with high accuracy (~97%)
- Uses NLP techniques and TF-IDF vectorization
- Clean and modern user interface
- End-to-end ML pipeline (training → deployment)
- REST API + Web UI
- Production-ready deployment

---

## 🧠 Tech Stack
- **Python**
- **Flask**
- **Scikit-learn**
- **NLTK**
- **Pandas, NumPy**
- **TF-IDF Vectorizer**
- **Naive Bayes Classifier**
- **HTML, CSS**
- **Render (Deployment)**

---

## 📊 Dataset
- **SMS Spam Collection Dataset**
- Labels:
  - `spam` → Spam message
  - `ham` → Not Spam

---

## ⚙️ How It Works
1. Email text is cleaned using NLP preprocessing
2. Text is converted into numerical features using TF-IDF
3. A Naive Bayes model predicts spam or not spam
4. Result is displayed on the web interface

