import pandas as pd
import os
from typing import List, Dict

class RAGService:
    def __init__(self, dataset_path: str, top_k: int = 3):
        self.dataset_path = dataset_path
        self.top_k = top_k
        self.df = None

    def load(self):
        if not os.path.exists(self.dataset_path):
            print(f"⚠️ RAG Warning: Dataset not found at {self.dataset_path}. RAG will be disabled.")
            return
        
        try:
            self.df = pd.read_csv(self.dataset_path)
            print(f"✓ RAG Service loaded: {len(self.df)} cases from {os.path.basename(self.dataset_path)}")
        except Exception as e:
            print(f"❌ RAG Error loading dataset: {e}")

    def query(self, user_prompt: str) -> str:
        """
        Tìm kiếm các trường hợp tương tự trong dataset và trả về ngữ cảnh (context).
        Ở bản v1, ta dùng keyword matching đơn giản trên cột 'prompt'.
        """
        if self.df is None or self.df.empty:
            return ""

        # Keyword matching đơn giản (có thể nâng cấp lên Vector Search sau)
        keywords = user_prompt.lower().split()
        
        # Tìm các dòng có chứa keyword
        mask = self.df['prompt'].str.contains('|'.join(keywords), case=False, na=False)
        similar_cases = self.df[mask].head(self.top_k)

        if similar_cases.empty:
            # Nếu không tìm thấy, lấy ngẫu nhiên 2 mẫu để Gemini tham khảo format
            similar_cases = self.df.sample(min(2, len(self.df)))

        context = "### HISTORICAL REFERENCE CASES (RAG)\n"
        for _, row in similar_cases.iterrows():
            context += f"- PROMPT: {row['prompt']}\n"
            context += f"  ISSUES: {row['issues']}\n"
            context += f"  ACTIONS: {row['actions']}\n"
            context += f"  REFINED: {row['refined_prompt']}\n\n"
        
        return context
