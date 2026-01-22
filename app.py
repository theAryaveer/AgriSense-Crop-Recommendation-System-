from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import joblib
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re
import requests

# Load environment variables
load_dotenv()

# Configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'crop-recommendation-secret-key-2024')
CORS(app)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crop_recommendations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Global variables for model and dictionary
crop_model = None
crop_dict = None
scaler = None
crop_info = {}  # Crop information for advanced analysis

# Bilingual crop names
CROP_TRANSLATIONS = {
    'rice': {'en': 'Rice', 'hi': 'चावल'},
    'wheat': {'en': 'Wheat', 'hi': 'गेहूं'},
    'maize': {'en': 'Maize', 'hi': 'मक्का'},
    'chickpea': {'en': 'Chickpea', 'hi': 'चना'},
    'kidneybeans': {'en': 'Kidney Beans', 'hi': 'राजमा'},
    'pigeonpeas': {'en': 'Pigeon Peas', 'hi': 'अरहर'},
    'mothbeans': {'en': 'Moth Beans', 'hi': 'मोठ'},
    'mungbean': {'en': 'Mung Bean', 'hi': 'मूंग'},
    'blackgram': {'en': 'Black Gram', 'hi': 'उड़द'},
    'lentil': {'en': 'Lentil', 'hi': 'मसूर'},
    'pomegranate': {'en': 'Pomegranate', 'hi': 'अनार'},
    'banana': {'en': 'Banana', 'hi': 'केला'},
    'mango': {'en': 'Mango', 'hi': 'आम'},
    'grapes': {'en': 'Grapes', 'hi': 'अंगूर'},
    'watermelon': {'en': 'Watermelon', 'hi': 'तरबूज'},
    'muskmelon': {'en': 'Muskmelon', 'hi': 'खरबूजा'},
    'apple': {'en': 'Apple', 'hi': 'सेब'},
    'orange': {'en': 'Orange', 'hi': 'संतरा'},
    'papaya': {'en': 'Papaya', 'hi': 'पपीता'},
    'coconut': {'en': 'Coconut', 'hi': 'नारियल'},
    'cotton': {'en': 'Cotton', 'hi': 'कपास'},
    'jute': {'en': 'Jute', 'hi': 'जूट'},
    'coffee': {'en': 'Coffee', 'hi': 'कॉफ़ी'}
}

# ============= DATABASE MODELS =============

class User(db.Model):
    """User model - stores farmer information"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to predictions
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'city': self.city,
            'phone': self.phone,
            'created_at': self.created_at.isoformat()
        }

class Prediction(db.Model):
    """Prediction model - stores crop prediction history"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    soil_data = db.Column(db.String(500), nullable=False)  # JSON string of N, P, K, etc.
    predicted_crop = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    results_json = db.Column(db.Text, nullable=True)  # Complete top 3 results with analysis
    date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'soil_data': json.loads(self.soil_data) if self.soil_data else {},
            'predicted_crop': self.predicted_crop,
            'confidence': self.confidence,
            'results_json': json.loads(self.results_json) if self.results_json else None,
            'date': self.date.isoformat()
        }

# Initialize database tables
with app.app_context():
    db.create_all()
    print("✓ Database tables created successfully")

# Load models
def load_models():
    global crop_model, crop_dict, scaler
    try:
        dict_path = os.path.join('model', 'crop_dict.pkl')
        model_path = os.path.join('model', 'best_crop_model.pkl')  # Updated to match actual filename
        scaler_path = os.path.join('model', 'scaler.pkl')
        
        # Load crop dictionary
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                crop_dict = pickle.load(f)
            print(f"✓ Loaded crop dictionary with {len(crop_dict)} crops")
        else:
            print(f"✗ Warning: crop_dict.pkl not found at {dict_path}")
        
        # Load crop model using joblib
        if os.path.exists(model_path):
            crop_model = joblib.load(model_path)
            print(f"✓ Loaded crop prediction model")
        else:
            print(f"✗ Warning: crop_model.pkl not found at {model_path}")
            print("  Please place your trained model file in the 'model' directory")
        
        # Load scaler
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✓ Loaded feature scaler")
        else:
            print(f"✗ Warning: scaler.pkl not found at {scaler_path}")
            print("  The prediction pipeline requires a scaler for accurate results")
    
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")

# Configure Gemini AI
def configure_gemini():
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("✗ Warning: GEMINI_API_KEY not found in environment variables")
        print("  AI features will not work without an API key")
        print("  Get your key from: https://makersuite.google.com/app/apikey")
        return False
    
    try:
        genai.configure(api_key=api_key)
        print("✓ Gemini AI configured successfully")
        return True
    except Exception as e:
        print(f"✗ Error configuring Gemini: {str(e)}")
        return False

# Load crop information for advanced analysis
def load_crop_info():
    global crop_info
    try:
        info_path = 'crop_info.json'
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                crop_info = json.load(f)
            print(f"✓ Loaded crop information for {len(crop_info)} crops")
        else:
            print(f"✗ Warning: crop_info.json not found")
            crop_info = {}
    except Exception as e:
        print(f"✗ Error loading crop_info.json: {str(e)}")
        crop_info = {}

def get_real_weather(city):
    """Fetches real-time weather from OpenWeatherMap."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key or "place_your_key" in api_key:
        print("⚠️ Skipping Weather API: Key not set.")
        return None
        
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": round(data["main"]["temp"], 1),
                "humidity": round(data["main"]["humidity"], 1)
            }
        else:
            print(f"⚠️ Weather API Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Weather Connection Failed: {e}")
        return None

def calculate_advanced_analysis(crop_name_en, user_soil):
    """
    Calculate nutrient gap, fertilizer needs, and economics for a crop.
    
    Args:
        crop_name_en: English name of the crop (lowercase)
        user_soil: Dict with N, P, K values from user input
    
    Returns:
        Dict with gap, bags, and economics data
    """
    crop_lower = crop_name_en.lower()
    
    # Get crop info from database
    if crop_info and crop_lower in crop_info:
        info = crop_info[crop_lower]
        
        # Calculate nutrient gaps (target - user, minimum 0)
        gap_N = max(0, info['target_N'] - user_soil['N'])
        gap_P = max(0, info['target_P'] - user_soil['P'])
        gap_K = max(0, info['target_K'] - user_soil['K'])
        
        # Calculate fertilizer bags (50kg bags)
        # Urea: 46% N, SSP: 16% P, MOP: 60% K
        urea_bags = round((gap_N / 0.46) / 50, 1) if gap_N > 0 else 0
        ssp_bags = round((gap_P / 0.16) / 50, 1) if gap_P > 0 else 0
        mop_bags = round((gap_K / 0.60) / 50, 1) if gap_K > 0 else 0
        
        # Calculate economics
        yield_kg = info['yield']
        price_per_kg = info['price']
        revenue = yield_kg * price_per_kg
        
        return {
            'gap': {
                'N': round(gap_N, 1),
                'P': round(gap_P, 1),
                'K': round(gap_K, 1)
            },
            'fertilizer': {
                'bags': {
                    'urea': urea_bags,
                    'ssp': ssp_bags,
                    'mop': mop_bags
                }
            },
            'economics': {
                'yield': yield_kg,
                'price': price_per_kg,
                'revenue': revenue
            }
        }
    else:
        # Return empty analysis if crop not found
        return {
            'gap': {'N': 0, 'P': 0, 'K': 0},
            'fertilizer': {'bags': {'urea': 0, 'ssp': 0, 'mop': 0}},
            'economics': {'yield': 0, 'price': 0, 'revenue': 0}
        }

# Initialize models and Gemini
load_models()
load_crop_info()
gemini_configured = configure_gemini()

@app.route('/')
def onboarding():
    """Serve the onboarding/welcome page"""
    return render_template('onboarding.html')

@app.route('/start', methods=['POST'])
def start():
    """Start the recommendation process - create/find user in database"""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        city = data.get('city', '').strip()
        phone = data.get('phone', '').strip()
        language = data.get('language', 'en')
        
        # Validation
        if not name or not city:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'Name and city are required',
                    'hi': 'नाम और शहर आवश्यक हैं'
                }
            }), 400
        
        # Check for existing user
        user = None
        if phone:
            # First try to find by phone
            user = User.query.filter_by(phone=phone).first()
        
        if not user:
            # If not found by phone, try name + city combination
            user = User.query.filter_by(name=name, city=city).first()
        
        if not user:
            # Create new user
            user = User(name=name, city=city, phone=phone if phone else None)
            db.session.add(user)
            db.session.commit()
            print(f"✓ Created new user: {name} from {city}")
        else:
            print(f"✓ Found existing user: {user.name} (ID: {user.id})")
        
        # Store user info in session
        session['user_id'] = user.id
        session['farmer_name'] = user.name
        session['city'] = city  # Store city to avoid asking again
        session['language'] = language
        
        return jsonify({
            'success': True,
            'redirect': '/dashboard'
        })
    
    except Exception as e:
        print(f"✗ Error in /start: {str(e)}")
        return jsonify({
            'success': False,
            'error': {
                'en': f'Server error: {str(e)}',
                'hi': f'सर्वर त्रुटि: {str(e)}'
            }
        }), 500

@app.route('/dashboard')
def dashboard():
    """Serve the main dashboard page"""
    # Get farmer info from session or use defaults
    farmer_name = session.get('farmer_name', 'Farmer')
    language = session.get('language', 'en')
    
    return render_template('dashboard.html', 
                          farmer_name=farmer_name,
                          language=language)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze location/city using Gemini AI"""
    try:
        if not gemini_configured:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'AI service not configured. Please set GEMINI_API_KEY.',
                    'hi': 'AI सेवा कॉन्फ़िगर नहीं है। कृपया GEMINI_API_KEY सेट करें।'
                }
            }), 500
        
        data = request.get_json()
        location = data.get('location', '').strip()
        
        if not location:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'Please provide a location',
                    'hi': 'कृपया स्थान प्रदान करें'
                }
            }), 400
        
        # --- HYBRID AI LOGIC START ---
        real_weather = None
        if location:
            real_weather = get_real_weather(location)
            if real_weather:
                print(f"✅ Real Weather Found for {location}: {real_weather}")
        
        # Construct prompt dynamically based on weather availability
        prompt = ""
        if real_weather:
            # HYBRID PROMPT: Real Temp/Humidity provided, ask only for Soil
            prompt = f"""
            I have real-time weather data for {location}:
            - Temperature: {real_weather['temp']}°C
            - Humidity: {real_weather['humidity']}%
            
            Based on this location ({location}) and its typical climate/soil, estimate the following soil parameters:
            1. Nitrogen (N) content (kg/ha)
            2. Phosphorus (P) content (kg/ha)
            3. Potassium (K) content (kg/ha)
            4. pH level (0-14)
            5. Rainfall (mm) - typical annual or seasonal
            
            Return JSON in this EXACT format (do not include temperature/humidity in the estimation, I will use real values):
            {{
                "N": <value>,
                "P": <value>,
                "K": <value>,
                "ph": <value>,
                "rainfall": <value>
            }}
            """
        else:
            # FULL AI ESTIMATION (Fallback)
            prompt = f"""
            Estimate the typical soil and environmental parameters for {location}.
            Return JSON in this EXACT format:
            {{
                "N": <value>,
                "P": <value>,
                "K": <value>,
                "temperature": <value>,
                "humidity": <value>,
                "ph": <value>,
                "rainfall": <value>
            }}
            """
        # --- HYBRID AI LOGIC END ---

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = response.text
        
        # Clean up JSON
        text = re.sub(r'```json\s*|\s*```', '', text)
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = text[start_idx:end_idx+1]
            estimated_data = json.loads(json_str)
            
            # --- HYBRID MERGE ---
            if real_weather:
                # Inject real data into the result
                estimated_data['temperature'] = real_weather['temp']
                estimated_data['humidity'] = real_weather['humidity']
                estimated_data['weather_source'] = 'live'
            else:
                estimated_data['weather_source'] = 'ai'
            
            return jsonify({
                'success': True,
                'data': estimated_data
            })
        else:
            raise ValueError("Invalid JSON from Gemini")
    
    except Exception as e:
        print(f"Gemini Error: {str(e)}")  # Terminal logging
        return jsonify({
            'success': False,
            'error': {
                'en': f'AI Error: {str(e)}',
                'hi': 'AI त्रुटि'
            }
        }), 500

@app.route('/estimate', methods=['POST'])
def estimate():
    """Convert farmer-friendly questionnaire responses to agricultural parameters"""
    try:
        if not gemini_configured:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'AI service not configured. Please set GEMINI_API_KEY.',
                    'hi': 'AI सेवा कॉन्फ़िगर नहीं है। कृपया GEMINI_API_KEY सेट करें।'
                }
            }), 500
        
        data = request.get_json()
        location = data.get('location', '').strip()
        soil_type = data.get('soil_type', '')
        water_availability = data.get('water_availability', '')
        season = data.get('season', '')
        fertilizer_usage = data.get('fertilizer_usage', '')
        temperature = data.get('temperature', '')
        
        if not location:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'Please provide a location',
                    'hi': 'कृपया स्थान प्रदान करें'
                }
            }), 400
        
        # Use Gemini to convert farmer-friendly inputs to technical parameters
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )
        
        prompt = f"""
        You are an agricultural expert for India. Based on these farmer-provided details, estimate realistic soil and climate parameters:
        
        Location: {location}
        Soil Type: {soil_type}
        Water Availability: {water_availability}
        Season: {season}
        Previous Fertilizer: {fertilizer_usage}
        Temperature Feel: {temperature}
        
        Convert these farmer-friendly responses into technical agricultural parameters.
        Return ONLY valid JSON with realistic values for India:
        {{
          "N": <nitrogen in kg/ha, range 0-140>,
          "P": <phosphorus in kg/ha, range 5-145>,
          "K": <potassium in kg/ha, range 5-205>,
          "temperature": <temperature in Celsius, range 10-50>,
          "humidity": <humidity in %, range 15-100>,
          "ph": <soil pH, range 3.5-9.5>,
          "rainfall": <annual rainfall in cm, range 20-300>
        }}
        
        Guidelines:
        - Black Soil: Higher N, neutral to slightly alkaline pH (7-8.5)
        - Red Soil: Lower N, acidic pH (5-6.5)
        - Sandy Soil: Lower nutrients, good drainage
        - Clay Soil: Higher nutrients, poor drainage
        - Loamy Soil: Balanced nutrients, good pH (6-7)
        - High Rainfall: Rainfall > 200cm, higher humidity
        - Moderate/Canal: Rainfall 100-200cm
        - Low Rain/Borewell: Rainfall < 100cm
        - Dry/Scarcity: Rainfall < 60cm
        - Urea: High N
        - DAP: High N and P
        - NPK: Balanced N, P, K
        - Organic: Moderate all, good pH
        - Kharif (Monsoon): June-Oct, high rainfall
        - Rabi (Winter): Nov-March, moderate temp
        - Zaid (Summer): March-June, high temp
        - Hot (30°C+): temperature 30-45
        - Moderate (20-30°C): temperature 20-30
        - Cold (<20°C): temperature 10-20
        
        Provide scientifically accurate estimates based on Indian agricultural conditions.
        """
        
        response = model.generate_content(prompt)
        
        # Parse JSON response
        try:
            parameters = json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback parsing
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                parameters = json.loads(json_match.group(0))
            else:
                raise ValueError("Invalid AI response format")
        
        # Validate parameters
        required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for param in required:
            if param not in parameters:
                return jsonify({
                    'success': False,
                    'error': {
                        'en': f'Missing parameter: {param}',
                        'hi': f'गायब पैरामीटर: {param}'
                    }
                }), 500
            # Convert to float
            parameters[param] = float(parameters[param])
        
        return jsonify({
            'success': True,
            'parameters': parameters,
            'location': location
        })
    
    except Exception as e:
        print(f"Gemini Estimate Error: {str(e)}")  # Terminal logging
        return jsonify({
            'success': False,
            'error': {
                'en': f'AI Error: {str(e)}',
                'hi': 'AI त्रुटि'
            }
        }), 500

@app.route('/weather', methods=['GET'])
def weather_only():
    """Get real-time weather for a city"""
    try:
        city = request.args.get('city', '').strip()
        if not city:
            return jsonify({'success': False, 'error': 'City required'}), 400
            
        real_weather = get_real_weather(city)
        if real_weather:
            return jsonify({
                'success': True,
                'data': real_weather,
                'source': 'live'
            })
        else:
            return jsonify({'success': False, 'error': 'Weather not found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict crop based on parameters"""
    try:
        if crop_model is None:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'Model not loaded.',
                    'hi': 'मॉडल लोड नहीं हुआ।'
                }
            }), 500
        
        if crop_dict is None:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'Crop dictionary not loaded.',
                    'hi': 'फसल शब्दकोश लोड नहीं हुआ।'
                }
            }), 500
        
        if scaler is None:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'Scaler not loaded. Cannot make accurate predictions.',
                    'hi': 'स्केलर लोड नहीं हुआ। सटीक भविष्यवाणी नहीं कर सकते।'
                }
            }), 500
        
        data = request.get_json()
        language = data.get('language', 'en')
        
        # Validate parameters
        required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing = [p for p in required if p not in data]
        
        if missing:
            return jsonify({
                'success': False,
                'error': {
                    'en': f'Missing: {", ".join(missing)}',
                    'hi': f'गायब: {", ".join(missing)}'
                }
            }), 400
        
        # Extract parameters
        try:
            N = float(data['N'])
            P = float(data['P'])
            K = float(data['K'])
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            ph = float(data['ph'])
            rainfall = float(data['rainfall'])
            weather_source = data.get('weather_source', 'ai')
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': {
                    'en': 'Parameters must be numeric',
                    'hi': 'पैरामीटर संख्यात्मक होने चाहिए'
                }
            }), 400
        
        # Create DataFrame for prediction
        features = pd.DataFrame({
            'N': [N], 'P': [P], 'K': [K],
            'temperature': [temperature], 'humidity': [humidity],
            'ph': [ph], 'rainfall': [rainfall]
        })
        
        # Apply scaler transformation
        features_scaled = scaler.transform(features)
        
        # Get prediction probabilities for all crops
        probabilities = crop_model.predict_proba(features_scaled)[0]
        
        # Get top 3 predictions
        top_3_indices = probabilities.argsort()[-3:][::-1]  # Sort and get top 3 in descending order
        
        # Build top prediction and alternatives with advanced analysis
        user_soil = {'N': N, 'P': P, 'K': K}
        top_predictions = []
        for idx in top_3_indices:
            crop_name_en = crop_dict.get(idx, f'Unknown (ID: {idx})')
            confidence = float(probabilities[idx] * 100)  # Convert to percentage
            
            # Translate crop name
            crop_lower = crop_name_en.lower()
            if crop_lower in CROP_TRANSLATIONS:
                crop_display = CROP_TRANSLATIONS[crop_lower].get(language, crop_name_en)
            else:
                crop_display = crop_name_en
            
            # Calculate advanced analysis (gap, fertilizer bags, economics)
            analysis = calculate_advanced_analysis(crop_name_en, user_soil)
            
            top_predictions.append({
                'crop': crop_display,
                'crop_en': crop_name_en,
                'confidence': round(confidence, 1),
                'analysis': analysis
            })
        
        # Prepare response with top prediction and alternatives
        response_data = {
            'success': True,
            'top_prediction': top_predictions[0],
            'alternatives': top_predictions[1:3] if len(top_predictions) > 1 else [],
            'soil_data': {
                'N': N,
                'P': P,
                'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall,
                'weather_source': weather_source
            }
        }
        
        # Save prediction to database
        prediction_id = None
        try:
            user_id = session.get('user_id')
            if user_id:
                # Create soil data JSON string
                soil_data_json = json.dumps({
                    'N': N, 'P': P, 'K': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                })
                
                # Create complete results JSON (top 3 with analysis)
                results_json_str = json.dumps({
                    'top_prediction': top_predictions[0],
                    'alternatives': top_predictions[1:3] if len(top_predictions) > 1 else [],
                    'soil_data': response_data['soil_data']
                })
                
                # Save prediction
                prediction = Prediction(
                    user_id=user_id,
                    soil_data=soil_data_json,
                    predicted_crop=top_predictions[0]['crop_en'],
                    confidence=top_predictions[0]['confidence'],
                    results_json=results_json_str
                )
                db.session.add(prediction)
                db.session.commit()
                prediction_id = prediction.id
                print(f"✓ Saved prediction #{prediction_id} for user {user_id}: {top_predictions[0]['crop_en']} ({top_predictions[0]['confidence']}%)")
        except Exception as save_error:
            print(f"⚠ Warning: Could not save prediction to database: {str(save_error)}")
            # Don't fail the entire request if saving fails
        
        # Add prediction_id to response
        if prediction_id:
            response_data['prediction_id'] = prediction_id
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': {
                'en': f'Prediction error: {str(e)}',
                'hi': f'भविष्यवाणी त्रुटि: {str(e)}'
            }
        }), 500

@app.route('/history', methods=['GET'])
def history():
    """Get prediction history for current user"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': {
                    'en': 'User not logged in',
                    'hi': 'उपयोगकर्ता लॉग इन नहीं है'
                }
            }), 401
        
        # Get last 5 predictions for this user
        predictions = Prediction.query.filter_by(user_id=user_id)\
            .order_by(Prediction.date.desc())\
            .limit(5)\
            .all()
        
        return jsonify({
            'success': True,
            'history': [p.to_dict() for p in predictions]
        })
    
    except Exception as e:
        print(f"✗ Error in /history: {str(e)}")
        return jsonify({
            'success': False,
            'error': {
                'en': f'Error retrieving history: {str(e)}',
                'hi': f'इतिहास पुनर्प्राप्त करने में त्रुटि: {str(e)}'
            }
        }), 500

@app.route('/report/<int:prediction_id>')
def report(prediction_id):
    # 1. Fetch prediction from DB
    prediction = Prediction.query.get_or_404(prediction_id)
    # Fetch User
    user = User.query.get(prediction.user_id)
    
    # 2. Parse complete results JSON (containing top 3 with analysis)
    results = json.loads(prediction.results_json) if prediction.results_json else None
    
    # 3. Fallback & Data Repair: Ensure analysis exists
    if not results:
        # Construct basic data if missing
        results = {
            'top_prediction': {'crop': prediction.predicted_crop, 'crop_en': prediction.predicted_crop, 'confidence': prediction.confidence},
            'alternatives': [],
            'soil_data': json.loads(prediction.soil_data)
        }
    
    # 4. ROBUST DATA PATCHING: Ensure 'analysis' and 'fertilizer' exist for top prediction
    if 'analysis' not in results['top_prediction'] or 'fertilizer' not in results['top_prediction'].get('analysis', {}):
        # Regenerate analysis on the fly
        user_soil = {k: results['soil_data'].get(k, 0) for k in ['N', 'P', 'K']}
        results['top_prediction']['analysis'] = calculate_advanced_analysis(results['top_prediction']['crop_en'], user_soil)
        
    # 5. ROBUST DATA PATCHING: Ensure for alternatives too
    for alt in results.get('alternatives', []):
         if 'analysis' not in alt or 'fertilizer' not in alt.get('analysis', {}):
             user_soil = {k: results['soil_data'].get(k, 0) for k in ['N', 'P', 'K']}
             alt['analysis'] = calculate_advanced_analysis(alt.get('crop_en', alt['crop']), user_soil)

    # 6. Render template with full results
    return render_template('report.html', prediction=prediction, results=results, crop_info=crop_info, user=user)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌾 Enhanced Farmer-First Crop Recommendation System 🌾")
    print("="*70)
    print(f"\nFeatures:")
    print("  ✓ Bilingual Support (English/Hindi)")
    print("  ✓ Farmer Onboarding")
    print("  ✓ City-based AI Analysis")
    print("  ✓ Manual Parameter Input")
    print(f"\nStarting Flask server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
