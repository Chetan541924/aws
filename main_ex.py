import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env values
load_dotenv()

# Import the executions router (contains execute-with-madl)
from app.routers import executions


app = FastAPI(
    title="MADL Executor Test Server",
    description="Minimal server to test /testcases/{id}/execute-with-madl",
    version="1.0.0"
)

# Allow all origins for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register ONLY the executions router
app.include_router(
    executions.router,
    prefix="/executor",
    tags=["Executor"]
)


@app.get("/")
def root():
    return {"status": "OK", "message": "Executor-with-madl test server running"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
