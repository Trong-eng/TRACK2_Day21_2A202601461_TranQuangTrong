import os

import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Wine Quality Prediction API")

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model() -> None:
    """Tai model moi nhat tu S3 ve EC2 khi API khoi dong."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Boto3 tu dong dung IAM role gan voi EC2, khong can Access Key tren VM.
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print(f"Downloaded s3://{S3_BUCKET}/{S3_MODEL_KEY} to {MODEL_PATH}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health() -> dict[str, str]:
    """Xac nhan API va model da khoi dong thanh cong."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, int | str]:
    """Du doan nhan chat luong ruou tu dung 12 dac trung dau vao."""
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)",
        )

    prediction = int(model.predict([req.features])[0])
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}

    return {
        "prediction": prediction,
        "label": labels[prediction],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
