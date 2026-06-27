from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship, DeclarativeBase
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import pickle, joblib, numpy as np, pandas as pd
import os, json, re, requests
from dotenv import load_dotenv
import google.generativeai as genai
from contextlib import asynccontextmanager

# ============================================================
# LOAD ENV
# ============================================================

load_dotenv()

# ============================================================
# DATABASE SETUP
# ============================================================

DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}"
    f"/{os.getenv('DB_NAME')}"
)

engine       = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

# ============================================================
# DATABASE MODELS
# ============================================================

class User(Base):
    __tablename__ = "users"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(100), nullable=False)
    city        = Column(String(100), nullable=False)
    phone       = Column(String(20),  nullable=True)
    created_at  = Column(DateTime,    default=datetime.utcnow)
    predictions = relationship("Prediction", back_populates="user")

    def to_dict(self):
        return {
            "id":         self.id,
            "name":       self.name,
            "city":       self.city,
            "phone":      self.phone,
            "created_at": self.created_at.isoformat(),
        }


class Prediction(Base):
    __tablename__ = "predictions"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    soil_data      = Column(Text,      nullable=False)
    predicted_crop = Column(String(50), nullable=False)
    confidence     = Column(Float,      nullable=False)
    results_json   = Column(Text,      nullable=True)
    date           = Column(DateTime,  default=datetime.utcnow)
    user           = relationship("User", back_populates="predictions")

    def to_dict(self):
        return {
            "id":             self.id,
            "soil_data":      json.loads(self.soil_data)    if self.soil_data    else {},
            "predicted_crop": self.predicted_crop,
            "confidence":     self.confidence,
            "results_json":   json.loads(self.results_json) if self.results_json else None,
            "date":           self.date.isoformat(),
        }

# ============================================================
# CONSTANTS
# ============================================================

CROP_TRANSLATIONS = {
    "rice":        {"en": "Rice",         "hi": "चावल"},
    "wheat":       {"en": "Wheat",        "hi": "गेहूं"},
    "maize":       {"en": "Maize",        "hi": "मक्का"},
    "chickpea":    {"en": "Chickpea",     "hi": "चना"},
    "kidneybeans": {"en": "Kidney Beans", "hi": "राजमा"},
    "pigeonpeas":  {"en": "Pigeon Peas",  "hi": "अरहर"},
    "mothbeans":   {"en": "Moth Beans",   "hi": "मोठ"},
    "mungbean":    {"en": "Mung Bean",    "hi": "मूंग"},
    "blackgram":   {"en": "Black Gram",   "hi": "उड़द"},
    "lentil":      {"en": "Lentil",       "hi": "मसूर"},
    "pomegranate": {"en": "Pomegranate",  "hi": "अनार"},
    "banana":      {"en": "Banana",       "hi": "केला"},
    "mango":       {"en": "Mango",        "hi": "आम"},
    "grapes":      {"en": "Grapes",       "hi": "अंगूर"},
    "watermelon":  {"en": "Watermelon",   "hi": "तरबूज"},
    "muskmelon":   {"en": "Muskmelon",    "hi": "खरबूजा"},
    "apple":       {"en": "Apple",        "hi": "सेब"},
    "orange":      {"en": "Orange",       "hi": "संतरा"},
    "papaya":      {"en": "Papaya",       "hi": "पपीता"},
    "coconut":     {"en": "Coconut",      "hi": "नारियल"},
    "cotton":      {"en": "Cotton",       "hi": "कपास"},
    "jute":        {"en": "Jute",         "hi": "जूट"},
    "coffee":      {"en": "Coffee",       "hi": "कॉफ़ी"},
}

# ============================================================
# GLOBAL STATE
# ============================================================

crop_model   = None
crop_dict    = None
scaler       = None
crop_info    = {}
gemini_model = None

# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class StartRequest(BaseModel):
    name:     str
    city:     str
    phone:    Optional[str] = None
    language: str = "en"


class PredictRequest(BaseModel):
    N:              float
    P:              float
    K:              float
    temperature:    float
    humidity:       float
    ph:             float
    rainfall:       float
    language:       str = "en"
    weather_source: str = "ai"


class EstimateRequest(BaseModel):
    location:           str
    soil_type:          Optional[str] = None
    water_availability: Optional[str] = None
    season:             Optional[str] = None
    fertilizer_usage:   Optional[str] = None
    temperature:        Optional[str] = None
    humidity:           Optional[str] = None


# ============================================================
# LIFESPAN — runs on startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Create MySQL tables
    Base.metadata.create_all(bind=engine)
    print("✓ MySQL tables created")

    # Load ML model files
    global crop_model, crop_dict, scaler, gemini_model, crop_info

    if os.path.exists("model/crop_dict.pkl"):
        crop_dict = pickle.load(open("model/crop_dict.pkl", "rb"))
        print(f"✓ Crop dictionary loaded ({len(crop_dict)} crops)")
    else:
        print("✗ crop_dict.pkl not found")

    if os.path.exists("model/best_crop_model.pkl"):
        crop_model = joblib.load("model/best_crop_model.pkl")
        print("✓ ML model loaded")
    else:
        print("✗ best_crop_model.pkl not found")

    if os.path.exists("model/scaler.pkl"):
        scaler = joblib.load("model/scaler.pkl")
        print("✓ Scaler loaded")
    else:
        print("✗ scaler.pkl not found")

    if os.path.exists("crop_info.json"):
        crop_info = json.load(open("crop_info.json", encoding="utf-8"))
        print(f"✓ Crop info loaded ({len(crop_info)} crops)")
    else:
        print("✗ crop_info.json not found")

    # Configure Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        print("✓ Gemini configured")
    else:
        print("✗ GEMINI_API_KEY not set — AI features disabled")

    yield

    print("App shutting down...")

# ============================================================
# APP INIT
# ============================================================

app = FastAPI(title="AgriSense API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ============================================================
# SESSION — simple in-memory (replace with Redis in production)
# ============================================================

sessions: dict = {}

def get_session(request: Request) -> dict:
    sid = request.cookies.get("session_id", "default")
    return sessions.setdefault(sid, {})

# ============================================================
# DATABASE DEPENDENCY
# ============================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================
# HELPERS
# ============================================================

def _require_gemini():
    if gemini_model is None:
        return {
            "en": "AI service not configured. Please set GEMINI_API_KEY.",
            "hi": "AI सेवा कॉन्फ़िगर नहीं है।",
        }
    return None


def _parse_gemini_json(text: str) -> dict:
    text  = re.sub(r"```json\s*|\s*```", "", text).strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in Gemini response: {text[:200]}")
    return json.loads(text[start:end + 1])


def _call_gemini(prompt: str) -> dict:
    response = gemini_model.generate_content(prompt)
    return _parse_gemini_json(response.text)


def get_real_weather(city: str):
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    if not api_key:
        print("⚠ Weather API key not set")
        return None
    try:
        r = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={api_key}&units=metric",
            timeout=5,
        )
        if r.status_code == 200:
            d = r.json()
            return {
                "temp":     round(d["main"]["temp"],     1),
                "humidity": round(d["main"]["humidity"], 1),
            }
        print(f"⚠ Weather API returned {r.status_code} for {city}")
    except Exception as e:
        print(f"⚠ Weather request failed: {e}")
    return None


def estimate_soil_params(
    location:     str,
    live_weather,
    manual_temp:  str = None,
    manual_humid: str = None,
    extra_context: str = "",
) -> dict:
    """
    Ask Gemini to estimate soil parameters.

    Priority for temperature and humidity —
    1. Farmer entered manually  → use that
    2. OpenWeather live data    → use that
    3. Nothing available        → ask Gemini to estimate
    """

    # Decide what temperature and humidity to use
    use_temp  = None
    use_humid = None
    weather_source = "ai"

    if manual_temp:
        # Farmer typed manually — highest priority
        use_temp       = float(manual_temp)
        use_humid      = float(manual_humid) if manual_humid else None
        weather_source = "manual"

    elif live_weather:
        # Live weather available — use it
        use_temp       = live_weather["temp"]
        use_humid      = live_weather["humidity"]
        weather_source = "live"

    # Build prompt based on what we know
    if use_temp is not None and use_humid is not None:
        # Temperature and humidity known — only ask Gemini for soil
        prompt = f"""
I have weather data for {location}:
- Temperature: {use_temp}°C
- Humidity: {use_humid}%

Based on this location and the following farmer information, estimate soil parameters:
{extra_context}

Return ONLY this JSON — no extra text:
{{
    "N": <nitrogen kg/ha>,
    "P": <phosphorus kg/ha>,
    "K": <potassium kg/ha>,
    "ph": <soil pH>,
    "rainfall": <rainfall mm>
}}
"""
    else:
        # Nothing known — ask Gemini for everything
        prompt = f"""
Estimate typical soil and climate parameters for {location}.
{extra_context}

Return ONLY this JSON — no extra text:
{{
    "N": <nitrogen kg/ha>,
    "P": <phosphorus kg/ha>,
    "K": <potassium kg/ha>,
    "temperature": <celsius>,
    "humidity": <percentage>,
    "ph": <soil pH>,
    "rainfall": <rainfall mm>
}}
"""

    params = _call_gemini(prompt)

    # Inject temperature and humidity if we have them
    if use_temp is not None:
        params["temperature"] = use_temp
    if use_humid is not None:
        params["humidity"] = use_humid

    params["weather_source"] = weather_source

    # Convert all values to float
    for key in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        params[key] = float(params[key])

    return params


def calculate_advanced_analysis(crop_name_en: str, user_soil: dict) -> dict:
    info = crop_info.get(crop_name_en.lower())
    if not info:
        return {
            "gap":        {"N": 0, "P": 0, "K": 0},
            "fertilizer": {"bags": {"urea": 0, "ssp": 0, "mop": 0}},
            "economics":  {"yield": 0, "price": 0, "revenue": 0},
        }

    gap_N = max(0.0, info["target_N"] - user_soil["N"])
    gap_P = max(0.0, info["target_P"] - user_soil["P"])
    gap_K = max(0.0, info["target_K"] - user_soil["K"])

    return {
        "gap": {
            "N": round(gap_N, 1),
            "P": round(gap_P, 1),
            "K": round(gap_K, 1),
        },
        "fertilizer": {
            "bags": {
                "urea": round((gap_N / 0.46) / 50, 1) if gap_N else 0,
                "ssp":  round((gap_P / 0.16) / 50, 1) if gap_P else 0,
                "mop":  round((gap_K / 0.60) / 50, 1) if gap_K else 0,
            }
        },
        "economics": {
            "yield":   info["yield"],
            "price":   info["price"],
            "revenue": info["yield"] * info["price"],
        },
    }


def translate_crop(crop_en: str, lang: str) -> str:
    return CROP_TRANSLATIONS.get(crop_en.lower(), {}).get(lang, crop_en)


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def onboarding(request: Request):
    return templates.TemplateResponse(
        "onboarding.html", {"request": request}
    )


@app.post("/start")
def start(body: StartRequest, request: Request, db: Session = Depends(get_db)):
    if not body.name.strip() or not body.city.strip():
        raise HTTPException(
            status_code=400,
            detail={"en": "Name and city are required", "hi": "नाम और शहर आवश्यक हैं"},
        )

    # Check if user already exists
    user = None
    if body.phone:
        user = db.query(User).filter(User.phone == body.phone).first()
    if not user:
        user = db.query(User).filter(
            User.name == body.name, User.city == body.city
        ).first()
    if not user:
        user = User(name=body.name, city=body.city, phone=body.phone or None)
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"✓ New user: {body.name} ({body.city})")
    else:
        print(f"✓ Existing user: {user.name} (ID {user.id})")

    # Save to session
    sid = request.cookies.get("session_id", "default")
    sessions[sid] = {
        "user_id":     user.id,
        "farmer_name": user.name,
        "city":        body.city,
        "language":    body.language,
    }

    return {"success": True, "redirect": "/dashboard"}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    sess = get_session(request)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request":     request,
            "farmer_name": sess.get("farmer_name", "Farmer"),
            "language":    sess.get("language", "en"),
            "session":     sess,
        },
    )


@app.get("/weather")
def weather_only(city: str):
    if not city.strip():
        raise HTTPException(status_code=400, detail="City required")
    data = get_real_weather(city)
    if data:
        return {"success": True, "data": data, "source": "live"}
    raise HTTPException(status_code=404, detail="Weather not found")


@app.post("/estimate")
def estimate(body: EstimateRequest):
    err = _require_gemini()
    if err:
        raise HTTPException(status_code=500, detail=err)

    if not body.location.strip():
        raise HTTPException(
            status_code=400,
            detail={"en": "Location required", "hi": "स्थान आवश्यक है"},
        )

    # Build extra context from questionnaire answers
    ctx_lines = [
        f"Soil type: {body.soil_type}"                    if body.soil_type           else "",
        f"Water availability: {body.water_availability}"  if body.water_availability  else "",
        f"Season: {body.season}"                          if body.season              else "",
        f"Previous fertilizer used: {body.fertilizer_usage}" if body.fertilizer_usage else "",
    ]

    # Add manual temperature and humidity to context if provided
    if body.temperature:
        ctx_lines.append(f"Target temperature: {body.temperature}°C")
    if body.humidity:
        ctx_lines.append(f"Target humidity: {body.humidity}%")

    extra = "\n".join(line for line in ctx_lines if line)

    try:
        # Get live weather — but only use if farmer did not enter manually
        live = get_real_weather(body.location)

        params = estimate_soil_params(
            location     = body.location,
            live_weather = live,
            manual_temp  = body.temperature,
            manual_humid = body.humidity,
            extra_context = extra,
        )

        return {"success": True, "parameters": params, "location": body.location}

    except Exception as e:
        print(f"✗ /estimate error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"en": f"AI Error: {e}", "hi": "AI त्रुटि"},
        )


@app.post("/predict")
def predict(body: PredictRequest, request: Request, db: Session = Depends(get_db)):

    # Check all ML files are loaded
    for label, obj in [
        ("Model",         crop_model),
        ("Crop dict",     crop_dict),
        ("Scaler",        scaler),
    ]:
        if obj is None:
            raise HTTPException(
                status_code=500,
                detail={"en": f"{label} not loaded", "hi": f"{label} लोड नहीं हुआ"},
            )

    # Collect all 7 input values
    values = {
        "N":           body.N,
        "P":           body.P,
        "K":           body.K,
        "temperature": body.temperature,
        "humidity":    body.humidity,
        "ph":          body.ph,
        "rainfall":    body.rainfall,
    }

    # Scale inputs and run ML model
    features_scaled = scaler.transform(pd.DataFrame([values]))
    probabilities   = crop_model.predict_proba(features_scaled)[0]
    top3_indices    = probabilities.argsort()[-3:][::-1]

    # Build top 3 predictions
    user_soil       = {k: values[k] for k in ["N", "P", "K"]}
    top_predictions = []

    for idx in top3_indices:
        crop_en    = crop_dict.get(idx, f"Unknown ({idx})")
        confidence = round(float(probabilities[idx]) * 100, 1)
        top_predictions.append({
            "crop":       translate_crop(crop_en, body.language),
            "crop_en":    crop_en,
            "confidence": confidence,
            "analysis":   calculate_advanced_analysis(crop_en, user_soil),
        })

    soil_data = {**values, "weather_source": body.weather_source}

    response_data = {
        "success":        True,
        "top_prediction": top_predictions[0],
        "alternatives":   top_predictions[1:],
        "soil_data":      soil_data,
    }

    # Save prediction to MySQL
    try:
        sess    = get_session(request)
        user_id = sess.get("user_id")

        if user_id:
            pred = Prediction(
                user_id        = user_id,
                soil_data      = json.dumps(values),
                predicted_crop = top_predictions[0]["crop_en"],
                confidence     = top_predictions[0]["confidence"],
                results_json   = json.dumps({
                    "top_prediction": top_predictions[0],
                    "alternatives":   top_predictions[1:],
                    "soil_data":      soil_data,
                }),
            )
            db.add(pred)
            db.commit()
            db.refresh(pred)
            response_data["prediction_id"] = pred.id
            print(f"✓ Saved prediction #{pred.id}: "
                  f"{top_predictions[0]['crop_en']} "
                  f"({top_predictions[0]['confidence']}%)")

    except Exception as e:
        print(f"⚠ Could not save prediction: {e}")

    return response_data


@app.get("/history")
def history(request: Request, db: Session = Depends(get_db)):
    sess    = get_session(request)
    user_id = sess.get("user_id")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={"en": "Not logged in", "hi": "लॉग इन नहीं है"},
        )

    preds = (
        db.query(Prediction)
        .filter(Prediction.user_id == user_id)
        .order_by(Prediction.date.desc())
        .limit(5)
        .all()
    )

    return {"success": True, "history": [p.to_dict() for p in preds]}


@app.get("/report/{prediction_id}", response_class=HTMLResponse)
def report(prediction_id: int, request: Request, db: Session = Depends(get_db)):

    prediction = db.query(Prediction).filter(
        Prediction.id == prediction_id
    ).first()

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    user    = db.query(User).filter(User.id == prediction.user_id).first()
    results = json.loads(prediction.results_json) if prediction.results_json else None

    if not results:
        results = {
            "top_prediction": {
                "crop":       prediction.predicted_crop,
                "crop_en":    prediction.predicted_crop,
                "confidence": prediction.confidence,
            },
            "alternatives": [],
            "soil_data":    json.loads(prediction.soil_data),
        }

    # Regenerate analysis if missing on old records
    user_soil = {k: results["soil_data"].get(k, 0) for k in ["N", "P", "K"]}
    for entry in [results["top_prediction"]] + results.get("alternatives", []):
        if "analysis" not in entry or "fertilizer" not in entry.get("analysis", {}):
            entry["analysis"] = calculate_advanced_analysis(
                entry.get("crop_en", entry["crop"]), user_soil
            )

    return templates.TemplateResponse(
        "report.html",
        {
            "request":    request,
            "prediction": prediction,
            "results":    results,
            "crop_info":  crop_info,
            "user":       user,
        },
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🌾  AgriSense — Crop Recommendation System")
    print("=" * 60)
    print("  http://127.0.0.1:5000   |   Press Ctrl+C to stop\n")
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)