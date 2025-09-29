"""
Main entry point for the Zen Music Analyzer API
This file is required for Railway's Railpack to automatically detect and start the FastAPI app
"""



import os
from realtime_embedding_api_fixed import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
