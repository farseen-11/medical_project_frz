from fastapi import FastAPI, Depends,HTTPException
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware
import src.models as models
import src.schemas as schemas
from src.database import engine, SessionLocal
from passlib.context import CryptContext
import joblib ,math,torch,json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.llmdoc import get_doctor_from_llm
import requests
import time
from rapidfuzz import fuzz


models.Base.metadata.create_all(bind=engine)


app = FastAPI()

pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto"
)

MODEL_PATH = r"C:\medical_project_frz-main\biobert_disease_topk_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
label_encoder = joblib.load(f"{MODEL_PATH}/label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open(r"C:\medical_project_frz-main\json_files\hospitals.json", "r", encoding="utf-8") as f:
    hospital_dataset = json.load(f)

print("Dataset loaded:", len(hospital_dataset))

def enrich_symptoms(text: str):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) > 1:
        return "Patient reports " + ", ".join(parts[:-1]) + " and " + parts[-1] + "."
    return "Patient reports " + text + "."

def predict_top_k(symptoms, k=5):
    inputs = tokenizer(
        symptoms,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    topk_idx = np.argsort(probs)[-k:][::-1]

    return [
        {
            "disease": label_encoder.inverse_transform([i])[0],
            "confidence": float(probs[i])
        }
        for i in topk_idx
    ]


def match_hospital(osm_name):
    best_match = None
    best_score = 0

    for h in hospital_dataset:
        score = fuzz.partial_ratio(
            osm_name.lower(),
            h.get("title", "").lower()
        )

        if score > best_score:
            best_score = score
            best_match = h


    if best_score > 75:
        return best_match
    return None


with open(r"C:\medical_project_frz-main\json_files\emergency_hospitals.json", "r", encoding="utf-8") as f:
    emergency_data = json.load(f)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)

    a = (
        math.sin(dLat/2)**2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dLon/2)**2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/patients/")
def create_patient(patient: schemas.PatientCreate, db: Session = Depends(get_db)):
    existing_patient = db.query(models.Patient).filter(
        models.Patient.name == patient.name , models.Patient.phone == patient.phone).first()
    if existing_patient:
        print("patient already exist")
        return {"message": "Patient already exists"}

    hashed_password = pwd_context.hash(patient.password)
    new_patient = models.Patient(
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        phone=patient.phone,
        email=patient.email,
        password_hash=hashed_password
    )
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    return {"message": "Patient registered successfully"}

@app.post("/signin/")
def verifying(user: schemas.Signing, db: Session = Depends(get_db)):
    exist_patient = db.query(models.Patient).filter(
        models.Patient.name == user.name , models.Patient.phone == user.phone).first()
    
    if not exist_patient:
        raise HTTPException(status_code=400, detail="User not exist")
    if not pwd_context.verify(user.password,exist_patient.password_hash ):
        raise HTTPException(status_code=400, detail="Incorrect password")
    return {"message":"login successfull"}
        

@app.post("/prediction")
def predict_disease(data: schemas.SymptomInput):
    text = enrich_symptoms(data.symptoms)
    results = predict_top_k(text, data.top_k)

    diseases = [item["disease"] for item in results]

    llm_result = get_doctor_from_llm(diseases)
    print(llm_result)

    return {
        "results": results,
        "doctor": llm_result.doctor,
        "urgency": llm_result.urgency,
        "reason": llm_result.reason
    }
@app.get("/nearby_hospitals")
def get_nearby_hospitals(lat: float, lon: float):

    hospitals = []

    for h in hospital_dataset:

        loc = h.get("location", {})

        if "lat" in loc and "lng" in loc:

            distance = haversine(lat, lon, loc["lat"], loc["lng"])

            h_copy = h.copy()

            # return same structure as before
            h_copy["lat"] = loc["lat"]
            h_copy["lon"] = loc["lng"]
            h_copy["distance"] = distance

            hospitals.append(h_copy)

    # sort nearest first
    hospitals.sort(key=lambda x: x["distance"])

    return {"hospitals": hospitals}



@app.get("/emergency_hospitals")
def get_emergency_hospitals(lat: float, lon: float):

    hospitals = []

    for h in emergency_data:
        loc = h.get("location", {})
        if "lat" in loc and "lng" in loc:

            dist = haversine(lat, lon, loc["lat"], loc["lng"])

            h_copy = h.copy()
            h_copy["distance"] = dist

            hospitals.append(h_copy)

    hospitals.sort(key=lambda x: x["distance"])

    return {"hospitals": hospitals}