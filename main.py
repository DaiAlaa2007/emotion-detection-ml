from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --------------------
# قراءة الداتا
# --------------------
data = pd.read_csv("data.csv")

texts = data["text"]
labels = data["emotion"]

# --------------------
# تحويل النص لأرقام
# --------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# --------------------
# تدريب الموديل
# --------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

# --------------------
# FastAPI
# --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    message: str

@app.post("/predict")
def predict(req: PredictRequest):
    vector = vectorizer.transform([req.message])
    probs = model.predict_proba(vector)[0]

    classes = model.classes_
    max_prob = max(probs)
    predicted_emotion = classes[probs.argmax()]

    # لو الموديل مش واثق
    if max_prob < 0.4:
        return {
            "status": "unknown",
            "message": "مش متأكدة الإحساس ده تابع لأنهي فئة",
            "options": list(classes)
        }

    return {
        "status": "ok",
        "emotion": predicted_emotion,
        "confidence": round(float(max_prob), 2)
    }
class FeedbackRequest(BaseModel):
    text: str
    emotion: str

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    df = pd.read_csv("data.csv")

    new_row = {
        "text": req.text,
        "emotion": req.emotion
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("data.csv", index=False)

    return {"status": "saved"}
