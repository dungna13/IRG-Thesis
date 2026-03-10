import uvicorn
from src.api.routes import app

if __name__ == "__main__":
    # Khởi động Production Server
    print("🚀 Starting IRG Multi-Agent Production Server [v2.1 - API Fixes]...")
    print("Checking configuration...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
