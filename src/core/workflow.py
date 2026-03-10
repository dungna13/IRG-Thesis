import re
from PIL import Image
from ..agents.expert_agent import ExpertAgent
from ..services.image_service import ImageService
from ..services.rag_service import RAGService

class IRGWorkflow:
    def __init__(self, expert_agent: ExpertAgent, image_service: ImageService, rag_service: RAGService):
        self.expert = expert_agent
        self.image_service = image_service
        self.rag = rag_service

    def parse_response(self, text: str) -> dict:
        """Bóc tách phản hồi từ Expert Agent."""
        data = {"issues": "none", "actions": "none", "refined_prompt": "", "full_text": text}
        
        diag_match = re.search(r"DIAGNOSIS:\s*(.*?)(?=ACTIONS:|$)", text, re.I | re.S)
        act_match = re.search(r"ACTIONS:\s*(.*?)(?=REFINED PROMPT:|$)", text, re.I | re.S)
        prompt_match = re.search(r"REFINED PROMPT:\s*(.*)", text, re.I | re.S)
        
        if diag_match: data["issues"] = diag_match.group(1).strip()
        if act_match: data["actions"] = act_match.group(1).strip()
        if prompt_match: data["refined_prompt"] = prompt_match.group(1).strip()
        
        return data

    def run_refinement(self, prompt: str, iterations: int = 2):
        """Luồng điều phối chính giữa các Agent."""
        results = []
        
        # 0. RAG: Lấy ngữ cảnh lịch sử
        print("Workflow: Retrieving historical context (RAG)...")
        rag_context = self.rag.query(prompt)
        
        # 1. Phân tích ban đầu
        print("Workflow: Starting initialization...")
        init_res = self.expert.analyze_initial_prompt(prompt, rag_context=rag_context)
        init_data = self.parse_response(init_res)
        
        # 2. Sinh ảnh gốc
        current_prompt = init_data["refined_prompt"] or prompt
        current_image = self.image_service.generate(current_prompt, seed=42)
        
        results.append({
            "iteration": 0,
            "image": current_image,
            "response": init_data
        })
        
        for i in range(1, iterations + 1):
            print(f"Workflow: Iteration {i}/{iterations}...")
            
            # Đo đạc feedback
            stats = self.image_service.get_stats(current_image)
            
            # Expert đưa ra quyết định
            refinement_res = self.expert.analyze_feedback(prompt, stats, i, rag_context=rag_context)
            refinement_data = self.parse_response(refinement_res)
            
            # Thực hiện tinh chỉnh
            current_prompt = refinement_data["refined_prompt"] or current_prompt
            current_image = self.image_service.refine(
                image=current_image, 
                prompt=current_prompt, 
                strength=0.35, 
                seed=42 + i
            )
            
            results.append({
                "iteration": i,
                "image": current_image,
                "response": refinement_data,
                "stats": stats
            })
            
        return results
