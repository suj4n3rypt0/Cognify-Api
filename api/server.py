import os
import uvicorn

# ======================================
# COGNIFY AI CONFIGURATION
# ======================================

API_KEY = "rqsty-sk-g6gmuhMqQ+S1M1j6veob5LBeN3mNKwXx2Hu8C3JC1ABvZzOLUOSzd5jl/Cg2FqbkwX56jCLi5GH2mGRe8f043GPU85pS/Ppl1VZdoY6W4AM="

MODEL = "google/gemma-4-31b-it"

# Make configuration available to app.py
os.environ["REQUESTY_API_KEY"] = API_KEY
os.environ["REQUESTY_MODEL"] = MODEL

# ======================================
# START SERVER
# ======================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir="api"
    )
