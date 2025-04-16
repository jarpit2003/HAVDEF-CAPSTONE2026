# HAVDEF - Hindi Audio-Visual Deepfake Defense

**HAVDEF** is a real-time deepfake voice detection system designed to combat the growing threat of AI-generated fraud calls in India. Focused on **Hinglish (Hindi + English)** conversations, HAVDEF uses machine learning techniques to detect synthetic voices and alert users during suspicious phone calls.

## 🌐 Project Goal

To safeguard users—especially in India—from voice-based impersonation scams by detecting and flagging AI-generated speech in real-time.

---

## 🧠 Features

- 🎙️ Real-time voice input analysis
- 🧾 Hinglish (Hindi + English) language support
- 🤖 Deepfake voice detection using trained AI models
- 📲 Mobile-friendly app interface (planned)
- 🔔 Instant alert system on potential fraud calls

---

## 🔧 Tech Stack

### 🧪 AI & Machine Learning
- **Python**
- **PyTorch / TensorFlow** – Model training and inference
- **Librosa** – Audio preprocessing and feature extraction
- **Scikit-learn** – Additional ML utilities
- **pyaudio / sounddevice** – Real-time audio capture

### 📊 NLP & Language Support
- **IndicNLP** or **iNLTK** – For Hindi and Hinglish text processing
- **TextBlob / SpaCy** – Sentiment or linguistic cues (if used)

### 📱 Mobile App (Planned)
- **Flutter** or **React Native** – Cross-platform mobile development
- **Firebase** – Backend and notifications (optional)

### 🧪 Deepfake Detection Techniques
- Voice embedding comparison (e.g., **x-vectors**, **ECAPA-TDNN**)
- Spectrogram analysis using CNNs or RNNs
- Classifier based on audio authenticity scores

---

## 🛠️ How It Works (High-Level)

1. Capture incoming voice stream during a phone call
2. Extract acoustic features (MFCCs, pitch, spectral roll-off, etc.)
3. Feed features into a trained deepfake classifier model
4. Compute a confidence score on voice authenticity
5. Alert the user if the voice is suspected to be AI-generated

---

## 🚀 Getting Started (Dev Setup)

```bash
git clone https://github.com/your-username/havdef.git
cd havdef

# Install dependencies
pip install -r requirements.txt

# Run demo or test audio detection
python detect_fake_audio.py --input sample_audio.wav

