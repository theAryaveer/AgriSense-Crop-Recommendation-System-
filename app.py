from flask import Flask, render_template, request, jsonify, session
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

# ============================================================
# SETUP
# ============================================================

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'crop-recommendation-secret-key-2024')
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crop_recommendations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ============================================================
# CONSTANTS
# ============================================================

CROP_TRANSLATIONS = {
    'rice':        {'en': 'Rice',         'hi': 'चावल'},
    'wheat':       {'en': 'Wheat',        'hi': 'गेहूं'},
    'maize':       {'en': 'Maize',        'hi': 'मक्का'},
    'chickpea':    {'en': 'Chickpea',     'hi': 'चना'},
    'kidneybeans': {'en': 'Kidney Beans', 'hi': 'राजमा'},
    'pigeonpeas':  {'en': 'Pigeon Peas',  'hi': 'अरहर'},
    'mothbeans':   {'en': 'Moth Beans',   'hi': 'मोठ'},
    'mungbean':    {'en': 'Mung Bean',    'hi': 'मूंग'},
    'blackgram':   {'en': 'Black Gram',   'hi': 'उड़द'},
    'lentil':      {'en': 'Lentil',       'hi': 'मसूर'},
    'pomegranate': {'en': 'Pomegranate',  'hi': 'अनार'},
    'banana':      {'en': 'Banana',       'hi': 'केला'},
    'mango':       {'en': 'Mango',        'hi': 'आम'},
    'grapes':      {'en': 'Grapes',       'hi': 'अंगूर'},
    'watermelon':  {'en': 'Watermelon',   'hi': 'तरबूज'},
    'muskmelon':   {'en': 'Muskmelon',    'hi': 'खरबूजा'},
    'apple':       {'en': 'Apple',        'hi': 'सेब'},
    'orange':      {'en': 'Orange',       'hi': 'संतरा'},
    'papaya':      {'en': 'Papaya',       'hi': 'पपीता'},
    'coconut':     {'en': 'Coconut',      'hi': 'नारियल'},
    'cotton':      {'en': 'Cotton',       'hi': 'कपास'},
    'jute':        {'en': 'Jute',         'hi': 'जूट'},
    'coffee':      {'en': 'Coffee',       'hi': 'कॉफ़ी'},
}

# ============================================================
# DATABASE MODELS
# ============================================================

class User(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(100), nullable=False)
    city        = db.Column(db.String(100), nullable=False)
    phone       = db.Column(db.String(20),  nullable=True)
    created_at  = db.Column(db.DateTime,    default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'city': self.city,
            'phone': self.phone, 'created_at': self.created_at.isoformat(),
        }


class Prediction(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    soil_data      = db.Column(db.String(500), nullable=False)
    predicted_crop = db.Column(db.String(50),  nullable=False)
    confidence     = db.Column(db.Float,        nullable=False)
    results_json   = db.Column(db.Text,         nullable=True)
    date           = db.Column(db.DateTime,     default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'soil_data':      json.loads(self.soil_data)    if self.soil_data    else {},
            'predicted_crop': self.predicted_crop,
            'confidence':     self.confidence,
            'results_json':   json.loads(self.results_json) if self.results_json else None,
            'date':           self.date.isoformat(),
        }


with app.app_context():
    db.create_all()
    print("✓ Database tables created successfully")

# ============================================================
# GLOBAL STATE  (loaded once at startup, never per-request)
# ============================================================

crop_model   = None
crop_dict    = None
scaler       = None
crop_info    = {}
gemini_model = None   # single shared Gemini instance


def load_models():
    global crop_model, crop_dict, scaler
    paths = {
        'dict':   os.path.join('model', 'crop_dict.pkl'),
        'model':  os.path.join('model', 'best_crop_model.pkl'),
        'scaler': os.path.join('model', 'scaler.pkl'),
    }
    if os.path.exists(paths['dict']):
        with open(paths['dict'], 'rb') as f:
            crop_dict = pickle.load(f)
        print(f"✓ Crop dictionary loaded ({len(crop_dict)} crops)")
    else:
        print(f"✗ crop_dict.pkl not found at {paths['dict']}")

    if os.path.exists(paths['model']):
        crop_model = joblib.load(paths['model'])
        print("✓ Crop prediction model loaded")
    else:
        print(f"✗ best_crop_model.pkl not found at {paths['model']}")

    if os.path.exists(paths['scaler']):
        scaler = joblib.load(paths['scaler'])
        print("✓ Feature scaler loaded")
    else:
        print(f"✗ scaler.pkl not found at {paths['scaler']}")


def load_crop_info():
    global crop_info
    if os.path.exists('crop_info.json'):
        with open('crop_info.json', 'r', encoding='utf-8') as f:
            crop_info = json.load(f)
        print(f"✓ Crop info loaded ({len(crop_info)} crops)")
    else:
        print("✗ crop_info.json not found")


def configure_gemini():
    """Create the single shared Gemini model instance. Returns True on success."""
    global gemini_model
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("✗ GEMINI_API_KEY not set – AI features disabled")
        return False
    try:
        genai.configure(api_key=api_key)
        # gemini-1.5-flash: free-tier compatible, fast, low token cost
        gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Gemini AI configured (gemini-2.5-flash)")
        return True
    except Exception as e:
        print(f"✗ Gemini configuration failed: {e}")
        return False


load_models()
load_crop_info()
gemini_configured = configure_gemini()

# ============================================================
# SHARED HELPERS
# ============================================================

def _require_gemini():
    """Return a bilingual error dict when Gemini is unavailable, else None."""
    if not gemini_configured or gemini_model is None:
        return {
            'en': 'AI service not configured. Please set GEMINI_API_KEY.',
            'hi': 'AI सेवा कॉन्फ़िगर नहीं है। कृपया GEMINI_API_KEY सेट करें।',
        }
    return None


def _parse_gemini_json(text: str) -> dict:
    """Strip markdown fences and return the first JSON object found in text."""
    text = re.sub(r'```json\s*|\s*```', '', text).strip()
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in Gemini response: {text[:200]}")
    return json.loads(text[start:end + 1])


def _call_gemini(prompt: str) -> dict:
    """Send a prompt to the shared Gemini model and return parsed JSON."""
    response = gemini_model.generate_content(prompt)
    return _parse_gemini_json(response.text)


def get_real_weather(city: str) -> dict | None:
    """Return live temp + humidity from OpenWeatherMap, or None on failure."""
    api_key = os.getenv('OPENWEATHER_API_KEY', '')
    if not api_key or 'place_your_key' in api_key:
        print("⚠ Weather API key not set – skipping live weather")
        return None
    try:
        url = (f"http://api.openweathermap.org/data/2.5/weather"
               f"?q={city}&appid={api_key}&units=metric")
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            d = r.json()
            return {'temp':     round(d['main']['temp'],     1),
                    'humidity': round(d['main']['humidity'], 1)}
        print(f"⚠ Weather API returned {r.status_code} for '{city}'")
    except Exception as e:
        print(f"⚠ Weather request failed: {e}")
    return None


def estimate_soil_params(location: str, live_weather: dict | None,
                         extra_context: str = '') -> dict:
    """
    Ask Gemini to estimate soil/climate parameters for a location.

    Two short prompt strategies to keep token usage low:
      - live_weather provided  → ask only for soil params (N/P/K/ph/rainfall),
                                  then inject real temp & humidity afterwards.
      - no live weather        → ask Gemini for all 7 parameters.

    extra_context adds farmer-supplied questionnaire answers (/estimate route).
    """
    if live_weather:
        # Hybrid prompt: real weather known, only ask for soil
        prompt = f"""
I have real-time weather data for {location}:
- Temperature: {live_weather['temp']}°C
- Humidity: {live_weather['humidity']}%

Based on this location and its typical soil, estimate:
1. Nitrogen (N) content (kg/ha)
2. Phosphorus (P) content (kg/ha)
3. Potassium (K) content (kg/ha)
4. pH level (0-14)
5. Rainfall (mm) - typical annual or seasonal
{extra_context}

Return JSON in this EXACT format (do not include temperature or humidity):
{{
    "N": <value>,
    "P": <value>,
    "K": <value>,
    "ph": <value>,
    "rainfall": <value>
}}
"""
    else:
        # Full AI estimation fallback
        prompt = f"""
Estimate the typical soil and environmental parameters for {location}.
{extra_context}

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

    params = _call_gemini(prompt)

    # Merge real weather into the result when available
    if live_weather:
        params['temperature']    = live_weather['temp']
        params['humidity']       = live_weather['humidity']
        params['weather_source'] = 'live'
    else:
        params['weather_source'] = 'ai'

    # Coerce all numeric fields to float
    for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
        params[key] = float(params[key])

    return params


def calculate_advanced_analysis(crop_name_en: str, user_soil: dict) -> dict:
    """Nutrient gap, fertilizer bags, and economics for a crop."""
    info = crop_info.get(crop_name_en.lower())
    if not info:
        return {
            'gap':        {'N': 0, 'P': 0, 'K': 0},
            'fertilizer': {'bags': {'urea': 0, 'ssp': 0, 'mop': 0}},
            'economics':  {'yield': 0, 'price': 0, 'revenue': 0},
        }

    gap_N = max(0.0, info['target_N'] - user_soil['N'])
    gap_P = max(0.0, info['target_P'] - user_soil['P'])
    gap_K = max(0.0, info['target_K'] - user_soil['K'])

    return {
        'gap': {'N': round(gap_N, 1), 'P': round(gap_P, 1), 'K': round(gap_K, 1)},
        'fertilizer': {
            'bags': {
                'urea': round((gap_N / 0.46) / 50, 1) if gap_N else 0,  # Urea  46 % N
                'ssp':  round((gap_P / 0.16) / 50, 1) if gap_P else 0,  # SSP   16 % P
                'mop':  round((gap_K / 0.60) / 50, 1) if gap_K else 0,  # MOP   60 % K
            }
        },
        'economics': {
            'yield':   info['yield'],
            'price':   info['price'],
            'revenue': info['yield'] * info['price'],
        },
    }


def translate_crop(crop_en: str, language: str) -> str:
    return CROP_TRANSLATIONS.get(crop_en.lower(), {}).get(language, crop_en)


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def onboarding():
    return render_template('onboarding.html')


@app.route('/start', methods=['POST'])
def start():
    try:
        data     = request.get_json()
        name     = data.get('name',     '').strip()
        city     = data.get('city',     '').strip()
        phone    = data.get('phone',    '').strip()
        language = data.get('language', 'en')

        if not name or not city:
            return jsonify({'success': False,
                            'error': {'en': 'Name and city are required',
                                      'hi': 'नाम और शहर आवश्यक हैं'}}), 400

        user = (User.query.filter_by(phone=phone).first() if phone else None)
        if not user:
            user = User.query.filter_by(name=name, city=city).first()
        if not user:
            user = User(name=name, city=city, phone=phone or None)
            db.session.add(user)
            db.session.commit()
            print(f"✓ New user: {name} ({city})")
        else:
            print(f"✓ Existing user: {user.name} (ID {user.id})")

        session.update({'user_id': user.id, 'farmer_name': user.name,
                        'city': city, 'language': language})
        return jsonify({'success': True, 'redirect': '/dashboard'})

    except Exception as e:
        print(f"✗ /start: {e}")
        return jsonify({'success': False,
                        'error': {'en': f'Server error: {e}',
                                  'hi': f'सर्वर त्रुटि: {e}'}}), 500


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
                           farmer_name=session.get('farmer_name', 'Farmer'),
                           language=session.get('language', 'en'))


@app.route('/analyze', methods=['POST'])
def analyze():
    """City → estimated soil/climate parameters (quick-estimate flow)."""
    err = _require_gemini()
    if err:
        return jsonify({'success': False, 'error': err}), 500

    data     = request.get_json()
    location = data.get('location', '').strip()
    if not location:
        return jsonify({'success': False,
                        'error': {'en': 'Please provide a location',
                                  'hi': 'कृपया स्थान प्रदान करें'}}), 400
    try:
        live_weather = get_real_weather(location)
        if live_weather:
            print(f"✅ Live weather for {location}: {live_weather}")
        params = estimate_soil_params(location, live_weather)
        return jsonify({'success': True, 'data': params})
    except Exception as e:
        print(f"✗ /analyze: {e}")
        return jsonify({'success': False,
                        'error': {'en': f'AI Error: {e}', 'hi': 'AI त्रुटि'}}), 500


@app.route('/estimate', methods=['POST'])
def estimate():
    """Farmer questionnaire answers → estimated soil/climate parameters."""
    err = _require_gemini()
    if err:
        return jsonify({'success': False, 'error': err}), 500

    data     = request.get_json()
    location = data.get('location', '').strip()
    if not location:
        return jsonify({'success': False,
                        'error': {'en': 'Please provide a location',
                                  'hi': 'कृपया स्थान प्रदान करें'}}), 400

    # Only include fields the farmer actually filled in
    ctx_lines = [
        f"Soil type: {data['soil_type']}"                   if data.get('soil_type')           else '',
        f"Water availability: {data['water_availability']}" if data.get('water_availability')   else '',
        f"Season: {data['season']}"                         if data.get('season')               else '',
        f"Previous fertilizer: {data['fertilizer_usage']}"  if data.get('fertilizer_usage')    else '',
        f"Temperature feel: {data['temperature']}"          if data.get('temperature')          else '',
    ]
    extra_context = '\n'.join(line for line in ctx_lines if line)

    try:
        live_weather = get_real_weather(location)
        params = estimate_soil_params(location, live_weather, extra_context)
        return jsonify({'success': True, 'parameters': params, 'location': location})
    except Exception as e:
        print(f"✗ /estimate: {e}")
        return jsonify({'success': False,
                        'error': {'en': f'AI Error: {e}', 'hi': 'AI त्रुटि'}}), 500


@app.route('/weather', methods=['GET'])
def weather_only():
    city = request.args.get('city', '').strip()
    if not city:
        return jsonify({'success': False, 'error': 'City required'}), 400
    data = get_real_weather(city)
    if data:
        return jsonify({'success': True, 'data': data, 'source': 'live'})
    return jsonify({'success': False, 'error': 'Weather not found'}), 404


@app.route('/predict', methods=['POST'])
def predict():
    """Run ML model and return top-3 crop predictions with full analysis."""
    for label, obj in [('Model', crop_model), ('Crop dictionary', crop_dict), ('Scaler', scaler)]:
        if obj is None:
            return jsonify({'success': False,
                            'error': {'en': f'{label} not loaded.',
                                      'hi': f'{label} लोड नहीं हुआ।'}}), 500

    data     = request.get_json()
    language = data.get('language', 'en')
    required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    missing  = [p for p in required if p not in data]
    if missing:
        return jsonify({'success': False,
                        'error': {'en': f'Missing: {", ".join(missing)}',
                                  'hi': f'गायब: {", ".join(missing)}'}}), 400
    try:
        values = {p: float(data[p]) for p in required}
    except (ValueError, TypeError):
        return jsonify({'success': False,
                        'error': {'en': 'Parameters must be numeric',
                                  'hi': 'पैरामीटर संख्यात्मक होने चाहिए'}}), 400

    try:
        features_scaled = scaler.transform(pd.DataFrame([values]))
        probabilities   = crop_model.predict_proba(features_scaled)[0]
        top3_indices    = probabilities.argsort()[-3:][::-1]

        user_soil       = {k: values[k] for k in ['N', 'P', 'K']}
        top_predictions = []
        for idx in top3_indices:
            crop_en    = crop_dict.get(idx, f'Unknown (ID: {idx})')
            confidence = round(float(probabilities[idx]) * 100, 1)
            top_predictions.append({
                'crop':       translate_crop(crop_en, language),
                'crop_en':    crop_en,
                'confidence': confidence,
                'analysis':   calculate_advanced_analysis(crop_en, user_soil),
            })

        soil_data     = {**values, 'weather_source': data.get('weather_source', 'ai')}
        response_data = {
            'success':        True,
            'top_prediction': top_predictions[0],
            'alternatives':   top_predictions[1:],
            'soil_data':      soil_data,
        }

        # Persist prediction to DB (non-fatal if it fails)
        try:
            user_id = session.get('user_id')
            if user_id:
                pred = Prediction(
                    user_id        = user_id,
                    soil_data      = json.dumps({k: values[k] for k in required}),
                    predicted_crop = top_predictions[0]['crop_en'],
                    confidence     = top_predictions[0]['confidence'],
                    results_json   = json.dumps({
                        'top_prediction': top_predictions[0],
                        'alternatives':   top_predictions[1:],
                        'soil_data':      soil_data,
                    }),
                )
                db.session.add(pred)
                db.session.commit()
                response_data['prediction_id'] = pred.id
                print(f"✓ Saved prediction #{pred.id}: "
                      f"{top_predictions[0]['crop_en']} ({top_predictions[0]['confidence']}%)")
        except Exception as save_err:
            print(f"⚠ Could not save prediction: {save_err}")

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'success': False,
                        'error': {'en': f'Prediction error: {e}',
                                  'hi': f'भविष्यवाणी त्रुटि: {e}'}}), 500


@app.route('/history', methods=['GET'])
def history():
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False,
                        'error': {'en': 'User not logged in',
                                  'hi': 'उपयोगकर्ता लॉग इन नहीं है'}}), 401
    try:
        preds = (Prediction.query
                 .filter_by(user_id=user_id)
                 .order_by(Prediction.date.desc())
                 .limit(5).all())
        return jsonify({'success': True, 'history': [p.to_dict() for p in preds]})
    except Exception as e:
        print(f"✗ /history: {e}")
        return jsonify({'success': False,
                        'error': {'en': f'Error retrieving history: {e}',
                                  'hi': f'इतिहास त्रुटि: {e}'}}), 500


@app.route('/report/<int:prediction_id>')
def report(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    user       = User.query.get(prediction.user_id)
    results    = json.loads(prediction.results_json) if prediction.results_json else None

    if not results:
        results = {
            'top_prediction': {'crop':       prediction.predicted_crop,
                               'crop_en':    prediction.predicted_crop,
                               'confidence': prediction.confidence},
            'alternatives': [],
            'soil_data':    json.loads(prediction.soil_data),
        }

    # Backwards-compatible: regenerate missing analysis blocks on old records
    user_soil = {k: results['soil_data'].get(k, 0) for k in ['N', 'P', 'K']}
    for entry in [results['top_prediction']] + results.get('alternatives', []):
        if 'analysis' not in entry or 'fertilizer' not in entry.get('analysis', {}):
            entry['analysis'] = calculate_advanced_analysis(
                entry.get('crop_en', entry['crop']), user_soil)

    return render_template('report.html', prediction=prediction,
                           results=results, crop_info=crop_info, user=user)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🌾  Crop Recommendation System")
    print("=" * 60)
    print("  http://127.0.0.1:5000   |   Press Ctrl+C to stop\n")
    app.run(debug=True, host='127.0.0.1', port=5000)