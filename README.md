🧠 AI Empowered Sentiment Intelligence
A modern Sentiment Analysis Web App powered by Machine Learning, featuring an animated futuristic UI with real-time confidence visualization. This project analyzes English text reviews and predicts whether the sentiment is Positive or Negative, along with a confidence score. 

## ✨ Features
🔍 Machine Learning
    - Trained sentiment classifier (English text)
    - Returns:
      - Sentiment: Positive / Negative
      - Confidence score (%)

## 🎨 Modern UI / UX
- Animated gradient background
- Moving AI grid & floating orbs
- Glassmorphism card
- Typing animation subtitle
- Emoji sentiment indicator 😄 / 😡
- Smooth progress bar confidence
- Mouse-based parallax effect

## ⚡ Tech Stack
- Frontend: React + Vite
- Backend: Flask + Flask-CORS
- ML: Scikit-learn
- Model Persistence: Pickle

## Code Structure
Project Sentiment Analysis/
│
├── backend/
│   ├── app.py          # Flask API
│   ├── train.py        # Train ML model
│   ├── model.pkl       # Saved ML model
│   ├── vectorizer.pkl  # Saved TF-IDF vectorizer
│
├── frontend/
│   ├── src/
│   │   └── App.jsx     # Main UI logic
│   ├── index.html
│   ├── package.json
│
└── README.md

## ⚙️ Requirements
**🧠 Backend (Python)**
_Make sure Python 3.7+ is installed._

**Install dependencies:**
_"pip install flask flask-cors scikit-learn numpy pandas"_

**🌐 Frontend (Node.js)**
Install:
1. Node.js (recommended v18+) 
2. npm

Check installation:
**_"node -v"_** And
**_"npm -v"_**

## 🚀 Cara Menjalankan Project
1) Backend Setup
   A. Masuk ke folder backend : "cd backend"

   B. Train model (cukup sekali saja) : "python train.py"
      Jika berhasil, akan muncul output seperti:
      - "Accuracy: 0.89xx"
      - "Model & vectorizer saved"
        
   C. Jalankan Flask API : "python app.py"
      Jika berhasil, akan muncul:
      "Sentiment Analysis API is running..."

     📍 Backend berjalan di:
      "http://127.0.0.1:5000"

2) Frontend Setup
Buka terminal baru (jangan tutup terminal backend yang sedang running).
A. Masuk ke folder frontend : "cd frontend"
B. Install dependencies : "npm install"
C. Jalankan frontend : "npm run dev"
D. Buka di browser
   "http://localhost:5173"

## 🧪 How to Use
1. Type an English sentence
2. Click Analyze
3. Result will show:
   - Sentiment emoji
   - Sentiment label
   - Confidence progress bar (%)

## 🛠 Troubleshooting
1. ❌ Backend Error / Fetch Failed
   - Make sure Flask is running
   - Check endpoint: "http://127.0.0.1:5000/predict"
   - Make sure Flask-CORS is installed
  
2. ❌ npm not recognized (Windows)
   - Run PowerShell as Administrator, then: "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
   - Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
  
3. ❌ Model Not Found

   Make sure:
   - model.pkl and vectorizer.pkl are inside /backend
   - train.py has been run before app.py
  
## 🌱 Future Improvements
   1. Neutral sentiment support
   2. Multi-language sentiment
   3. Model comparison (Naive Bayes vs SVM)
   4. Deployment (Vercel + Render)
   5. User history & analytics dashboard

## 👤 Author
Rizky Putra Ichfiandra,
AI / ML Enthusiast.
Built with ❤️, logic, dan begadang
