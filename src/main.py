from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ADD THIS LINE
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from api.routes import router
import uvicorn

app = FastAPI(
    title="Loan Recommendation API",
    description="Modular ML recommendation engine",
    version="2.0.0"
)

# ADD THIS ENTIRE BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1/loans", tags=["Loans"])

@app.get("/")
def root():
    return {
        "service": "ML Recommendation Platform",
        "version": "2.0.0",
        "engines": ["loans"],
        "docs": "/docs"
    }

if __name__ == "__main__":
    print("🚀 Starting Loan Recommendation API...")
    uvicorn.run(app, host="0.0.0.0", port=8080)