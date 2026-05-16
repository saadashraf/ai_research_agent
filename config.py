import os
from dotenv import load_dotenv

load_dotenv()

# --- Model settings ---
PROVIDER = "anthropic"
MODEL    = "claude-sonnet-4-5"

# --- Inference settings ---
MAX_TOKENS  = 4096
TEMPERATURE = 0.5   # 0 = deterministic, 1 = creative

# --- API keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")