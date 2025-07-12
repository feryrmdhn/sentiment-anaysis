from fastapi import FastAPI

app = FastAPI(
    title="Sentiment Analysis",
    description="API untuk analisis sentimen",
    version="1.0.0"
)

# default
@app.get("/")
async def health_check():
    return {"status": "ok"}