import os
import time
import google.generativeai as genai
from ..config import GeminiConfig

class GeminiService:
    def __init__(self, config: GeminiConfig):
        self.config = config
        self.model = None

    def load(self):
        api_key = self.config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or config.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config.model_name)
        print(f"✓ Gemini Service loaded: {self.config.model_name}")

    def generate(self, prompt: str, context: str = "") -> str:
        if not self.model:
            self.load()
            
        full_prompt = f"{context}\n\nUser Request: {prompt}" if context else prompt
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                # Nếu lỗi do vượt quá hạn mức (429), đợi lâu hơn một chút
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    wait_time = (attempt + 1) * 5 # Đợi 5s, 10s...
                    print(f"⚠️ API Busy (Rate Limit). Waiting {wait_time}s before retry {attempt+1}/{self.config.max_retries}...")
                    time.sleep(wait_time)
                    continue
                
                if attempt == self.config.max_retries - 1:
                    raise e
                time.sleep(self.config.retry_backoff * (attempt + 1))
        return ""

    def unload(self):
        self.model = None
