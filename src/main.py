from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # ADD THIS
from fastapi.staticfiles import StaticFiles  # ADD THIS
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files - ADD THIS
app.mount("/static", StaticFiles(directory="src"), name="static")

# Serve demo page - ADD THIS
@app.get("/demo")
async def get_demo():
    return FileResponse("src/demo.html")

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