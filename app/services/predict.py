from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import boto3
import pandas as pd
import io
import json
import os
from dotenv import load_dotenv
from app.utils.utils import load_vectorizer_from_s3

load_dotenv()

aws_region = os.getenv('AWS_REGION')

vectorizer = load_vectorizer_from_s3()

runtime_client = boto3.client('sagemaker-runtime', region_name=aws_region)

router = APIRouter(prefix="/v1", tags=["Prediction"])

class InputText(BaseModel):
    text: str

@router.post("/predict", description="Prediksi sentimen")
def predict_sentiment(data: InputText):
    sagemaker_endpoint = "linear-learner-sentiment-model-endpoint"

    try:
        # Preprocessing (TF-IDF)
        text = data.text
        tfidf_features = vectorizer.transform([text])

        # Convert fitur ke CSV string (tanpa header/index)
        csv_buffer = io.StringIO()
        if hasattr(tfidf_features, 'toarray'):
            features_array = tfidf_features.toarray()
        else:
            features_array = tfidf_features
        pd.DataFrame(features_array).to_csv(csv_buffer, header=False, index=False)
        input_csv = csv_buffer.getvalue()
        
        # Invoke SageMaker endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=sagemaker_endpoint,
            ContentType='text/csv',
            Body=input_csv
        )
        result = response['Body'].read().decode('utf-8')
        data_json = json.loads(result)
        pred_label = int(data_json['predictions'][0]['predicted_label'])
        
        # Ubah label ke string
        sentiment = "Positif" if pred_label == 1 else "Negatif"

        return {
            "status": "success",
            "input": text,
            "prediction": sentiment,
        }
    except Exception as e:
        print(f"Error saat prediksi: {e}")
        raise HTTPException(status_code=500, detail=f"Prediksi gagal: {e}")