# 🇯🇵 Sensei AI: Hybrid Kuzushiji Recognizer

A machine learning application designed to recognize handwritten Japanese characters (Kuzushiji/Hiragana) using a Hybrid AI approach. This project combines a custom **Local CNN model** trained on the KMNIST-49 dataset with **Google's Gemini 1.5 Flash API** to provide accurate predictions.

## 👨‍💻 Author Information

* **Name:** Abraham Shalom Nadeak
* **Student ID (NIM):** 10302240008
* **Major:** Software Engineering (RPL)

---

## 🚀 Project Overview

The **Kuzushiji Recognizer** solves the challenge of reading ancient or handwritten Japanese scripts. It uses a modular architecture separating the frontend (Streamlit) from the backend logic.

### Key Features
* **✍️ Interactive Canvas:** Draw characters directly on the screen for real-time prediction.
* **📂 Smart Image Upload:** Upload images with automatic aspect-ratio padding (prevents distortion).
* **🤖 Hybrid Intelligence:**
    * **Local Model:** A lightweight Convolutional Neural Network (CNN) trained on KMNIST-49 (Offline capable).
    * **Cloud Expert:** Integration with **Google Gemini 3 Flash Preview** via the official `google-genai` SDK for high-accuracy verification.
* **🛠 Modular Codebase:** Clean separation of concerns using a `modules/` package structure.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **Computer Vision:** OpenCV, Pillow, Numpy
* **Machine Learning:** TensorFlow / Keras
* **Cloud AI:** Google GenAI SDK (`google-genai`)

---