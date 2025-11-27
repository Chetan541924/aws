import os
import uvicorn
import logging
from fastapi import FastAPI
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("madl_api")

# Import your routers
from routers.reusable_methods import router as reusable_methods_router

# ------------------------------------------------------
#  INITIALIZE FASTAPI
# ------------------------------------------------------
app = FastAPI(
    title="MADL Reusable Method Engine",
    description="Semantic Search Engine for Test Step Reuse using Azure OpenAI + OpenSearch",
    version="1.0.0",
)

# ------------------------------------------------------
#  REGISTER ROUTERS
# ------------------------------------------------------
app.include_router(reusable_methods_router)

# ------------------------------------------------------
#  ROOT ENDPOINT
# ------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "OK",
        "message": "MADL Engine Running",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }

# ------------------------------------------------------
#  MAIN ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
