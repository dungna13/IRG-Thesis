from ..services.gemini_service import GeminiService

IRG_EXPERT_SYSTEM = (
    "Act: IRG Expert. Task: Analyze image stats (mean, std, max).\n"
    "Rules:\n"
    "- mean: <.28 Under, >.62 Over\n"
    "- std: <.12 Low-Con, >.28 High-Con\n"
    "- max: <.70 No-High, >.97 Blown-High\n"
    "Format (STRICT):\n"
    "DIAGNOSIS: [csv issues or 'none']\n"
    "ACTIONS:\n"
    "- [±% adjust]\n"
    "REFINED PROMPT: [improved prompt]"
)

class ExpertAgent:
    def __init__(self, gemini_service: GeminiService):
        self.gemini = gemini_service

    def analyze_initial_prompt(self, prompt: str, rag_context: str = "") -> str:
        msg = f"{rag_context}\nINIT REQUEST: {prompt}. Task: Phân tích bố cục/ánh sáng."
        return self.gemini.generate(prompt=msg, context=IRG_EXPERT_SYSTEM)

    def analyze_feedback(self, prompt: str, stats: dict, iteration: int, rag_context: str = "") -> str:
        s = f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, max={stats['max']:.4f}"
        msg = f"{rag_context}\nITER {iteration}. Prompt: {prompt}. Stats: {s}. Task: Fix issues & refine prompt."
        return self.gemini.generate(prompt=msg, context=IRG_EXPERT_SYSTEM)
