# Smart Crop Recommendation System

An AI-powered web application that recommends optimal crops based on soil and weather parameters.

## Features

- **Manual Input**: Enter 7 soil and weather parameters using interactive sliders
- **AI Consultant**: Describe your farm in natural language, and Gemini AI extracts the parameters
- **Instant Predictions**: Get crop recommendations using a trained Random Forest model (99.32% accuracy)
- **Modern UI**: Responsive, agriculture-themed dashboard with glassmorphism design

## Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))
- Trained model files:
  - `model/crop_model.pkl` (Random Forest model)
  - `model/crop_dict.pkl` (Label-to-crop mapping)

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "c:\Users\HP\Desktop\crop recommendation"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Open `.env` file
   - Replace `your_gemini_api_key_here` with your actual Gemini API key

4. **Ensure model files are in place**:
   ```
   crop recommendation/
   ├── model/
   │   ├── crop_model.pkl  ← Place your trained model here
   │   └── crop_dict.pkl   ← Label mapping dictionary
   ```

## Usage

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **Choose your input method**:
   - **Manual Input**: Use sliders to set N, P, K, temperature, humidity, pH, and rainfall
   - **AI Consultant**: Type a natural description like:
     > "I live in Punjab, soil is loamy with good nitrogen, temperature around 28°C, humid climate with monsoon rainfall of 150mm"

4. **Get your recommendation**: The system will predict the best crop for your conditions!

## Input Parameters

The model requires 7 parameters in this exact order:

1. **N** (Nitrogen): Soil nitrogen content ratio (0-1)
2. **P** (Phosphorus): Soil phosphorus content ratio (0-1)
3. **K** (Potassium): Soil potassium content ratio (0-1)
4. **Temperature**: Average temperature in Celsius (8-45°C)
5. **Humidity**: Relative humidity percentage (0-100%)
6. **pH**: Soil pH level (3-10)
7. **Rainfall**: Annual rainfall in millimeters (20-300mm)

## API Endpoints

### `GET /`
Returns the main dashboard HTML page.

### `POST /predict`
Accepts manual input parameters and returns crop prediction.

**Request Body**:
```json
{
  "N": 0.5,
  "P": 0.6,
  "K": 0.4,
  "temperature": 25,
  "humidity": 70,
  "ph": 6.5,
  "rainfall": 120
}
```

**Response**:
```json
{
  "success": true,
  "crop": "rice",
  "label": 20
}
```

### `POST /ai-analyze`
Accepts natural language description and returns structured parameters.

**Request Body**:
```json
{
  "description": "I farm in Rajasthan, hot climate, sandy soil..."
}
```

**Response**:
```json
{
  "success": true,
  "parameters": {
    "N": 0.45,
    "P": 0.38,
    "K": 0.52,
    "temperature": 35,
    "humidity": 40,
    "ph": 7.5,
    "rainfall": 60
  }
}
```

## Project Structure

```
crop recommendation/
├── app.py                  # Flask backend server
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (API keys)
├── README.md              # This file
├── model/
│   ├── crop_model.pkl     # Trained Random Forest model
│   └── crop_dict.pkl      # Crop label dictionary
└── templates/
    └── index.html         # Frontend dashboard
```

## Troubleshooting

### Model Not Loading
- Ensure `crop_model.pkl` exists in the `model/` directory
- Check file permissions

### AI Analysis Not Working
- Verify your Gemini API key is correct in `.env`
- Check internet connection
- Ensure you have API quota remaining

### Server Won't Start
- Check if port 5000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`

## Technologies Used

- **Backend**: Flask, Python
- **AI Integration**: Google Gemini 2.0 Flash
- **Frontend**: HTML5, Bootstrap 5, Vanilla JavaScript
- **ML Model**: Random Forest (scikit-learn)

## License

This project is for educational purposes.

## Support

For issues or questions, please check:
1. Model files are correctly placed
2. API key is valid and set in `.env`
3. All dependencies are installed
4. Python version is 3.8 or higher
