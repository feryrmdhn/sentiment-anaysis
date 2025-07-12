from fastapi import FastAPI
from app.services import predict

app = FastAPI(  
    title="Sentiment Analysis",
    description="API untuk analisis sentimen",
    version="1.0.0"
)

# default
@app.get("/")
async def health_check():
    return {"status": "ok"}

# predict
app.include_router(predict.router)
