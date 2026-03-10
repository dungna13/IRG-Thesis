"""
config.py — Cấu hình tập trung cho toàn bộ IRG pipeline.

Sử dụng Gemini API thay vì load model Qwen local.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv() # Load from .env file


@dataclass
class GeminiConfig:
    """Cấu hình Gemini API cho LLM inference."""

    # --- API ---
    api_key: str = ""  # Sẽ được lấy từ .env hoặc env var
    model_name: str = "gemini-3.1-flash-lite-preview" 

    # --- RAG ---
    rag_top_k: int = 5

    # --- Retry ---
    max_retries: int = 3
    retry_backoff: float = 1.0  # seconds, exponential backoff base

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY", "")


@dataclass
class ImageGenerationConfig:
    """Cấu hình cho SD image generation pipeline (Lightweight)."""

    # --- Stability AI API ---
    use_api: bool = True
    stability_api_key: str = field(default_factory=lambda: os.getenv("STABILITY_API_KEY", ""))
    engine_id: str = "stable-diffusion-xl-1024-v1-0" 

    # --- Tham số sinh ảnh ---
    image_height: int = 1024 # API hỗ trợ HD
    image_width: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    refiner_strength: float = 0.35

    # --- Seed ---
    seed_base: int = 42


@dataclass
class BenchmarkConfig:
    """Cấu hình cho benchmark & scoring."""

    # --- CLIP ---
    clip_model_name: str = "ViT-B/32"

    # --- Aesthetic ---
    aesthetic_model_path: str = ""  # Để trống sẽ dùng mặc định

    # --- Số variant so sánh ---
    variants: List[str] = field(
        default_factory=lambda: ["base_sdxl", "irg_2iter", "irg_3iter", "irg_4iter"]
    )


@dataclass
class PipelineConfig:
    """Cấu hình tổng hợp cho toàn bộ pipeline."""

    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    image_gen: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # --- Output ---
    output_dir: str = "./output"

    # --- Số vòng refinement (2, 3, hoặc 4) ---
    num_iterations: int = 2

    # --- RAG Dataset ---
    rag_dataset_file: str = "dataset_final_v3.csv"

    def load_prompts(self) -> List[str]:
        """Đọc danh sách prompts từ file."""
        if not os.path.exists(self.prompts_file):
            raise FileNotFoundError(f"Không tìm thấy file prompts: {self.prompts_file}")
        with open(self.prompts_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
