from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
from ..config import PipelineConfig
from ..services.gemini_service import GeminiService
from ..services.image_service import ImageService
from ..services.rag_service import RAGService
from ..agents.expert_agent import ExpertAgent
from ..core.workflow import IRGWorkflow

app = FastAPI(title="IRG-Thesis Multi-Agent API", version="2.1.0")

# Cấu hình Singletons
config = PipelineConfig()
gemini_service = GeminiService(config.gemini)
image_service = ImageService(config.image_gen)
rag_service = RAGService(config.rag_dataset_file) # NEW
expert_agent = ExpertAgent(gemini_service)
workflow = IRGWorkflow(expert_agent, image_service, rag_service) # Updated

# Khởi tạo models (Lazy loading trong production)
# image_service.load()

class RefinementRequest(BaseModel):
    prompt: str
    iterations: Optional[int] = 2

class RefinementResponse(BaseModel):
    request_id: str
    status: str
    message: str

@app.on_event("startup")
async def startup_event():
    gemini_service.load()
    image_service.load()
    rag_service.load() # NEW

@app.get("/")
def read_root():
    return {"status": "online", "model": config.image_gen.sd_model_id}

@app.post("/refine", response_model=RefinementResponse)
async def start_refinement(request: RefinementRequest):
    """
    Endpoint khởi chạy quy trình Agentic Refinement.
    """
    request_id = str(uuid.uuid4())
    # Trong production thật, ta dùng Celery/Redis. Ở đây giả lập chạy trực tiếp.
    try:
        results = workflow.run_refinement(request.prompt, iterations=request.iterations)
        
        # Lưu kết quả
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        for res in results:
            res["image"].save(f"{output_dir}/{request_id}_iter_{res['iteration']}.png")
            
        return {
            "request_id": request_id,
            "status": "completed",
            "message": f"Successfully completed {len(results)-1} iterations. Images saved to {output_dir}/"
        }
    except Exception as e:
        print(f"❌ Error in /refine: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
