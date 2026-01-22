# 🚀 Quick Start Guide

## What You Have Now

✅ Complete Flask backend (`app.py`)
✅ Beautiful Bootstrap frontend (`templates/index.html`)
✅ All dependencies installed
✅ Configuration files ready (`.env`, `requirements.txt`)
✅ Comprehensive documentation (`README.md`)

## ⚠️ Before You Can Run

You need 2 things:

### 1️⃣ Your Trained Model File
Place `crop_model.pkl` in the `model/` folder:
```
crop recommendation/
└── model/
    ├── crop_dict.pkl  ✅ (already here)
    └── crop_model.pkl  ⚠️ (you need to add this)
```

### 2️⃣ Gemini API Key (for AI feature)

1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key
4. Open `.env` file in this folder
5. Replace `your_gemini_api_key_here` with your actual key

Example:
```env
GEMINI_API_KEY=AIzaSyDxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## 🎯 How to Run

Once you have the model and API key:

```bash
# Navigate to the project
cd "c:\Users\HP\Desktop\crop recommendation"

# Run the server
python app.py
```

You'll see:
```
============================================================
🌾 Smart Crop Recommendation System 🌾
============================================================

✓ Loaded crop dictionary with 22 crops
✓ Loaded crop prediction model
✓ Gemini AI configured successfully

Starting Flask server at http://127.0.0.1:5000
```

Then open your browser to: **http://127.0.0.1:5000**

## 🎨 How to Use

### Option 1: Manual Input
1. Use the sliders to set your soil and weather values
2. Click "Predict Crop"
3. See the recommended crop!

### Option 2: AI Consultant
1. Type a description like:
   > "I farm in Punjab, soil is loamy, temperature 28°C, humid climate, 150mm rain"
2. Click "Analyze with AI"
3. Watch the sliders auto-fill
4. Get your crop recommendation!

## ❓ Troubleshooting

**"Model not loaded" error**
→ Make sure `crop_model.pkl` is in the `model/` folder

**"AI service not configured" error**
→ Add your Gemini API key to `.env` file

**Server won't start**
→ Run `pip install -r requirements.txt` again

## 📖 More Information

See [README.md](file:///c:/Users/HP/Desktop/crop%20recommendation/README.md) for full documentation!
