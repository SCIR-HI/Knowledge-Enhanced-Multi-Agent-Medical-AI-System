"""Central configuration for the medical multi-agent system."""

# OpenAI-compatible LLM service.
SERVE_URL = "http://localhost:8000/v1"
OPENAI_API_KEY = "not-needed"
MODEL_NAME = "Qwen3-32B-AWQ"

# Local retrieval resources. Replace these paths with downloaded local assets.
EMBEDDING_MODEL = "/path/to/bge-m3"
RERANK_MODEL = "/path/to/bge-reranker-base"
FAISS_INDEX_PATH = "/path/to/faiss_index_A"

# Multiple FAISS index versions can be registered here when available.
DEFAULT_FAISS_VERSION = "v4"
FAISS_INDEX_VERSIONS = {
    "v1": "/path/to/faiss_index_A_v1",
    "v2": "/path/to/faiss_index_A_v2",
    "v3": "/path/to/faiss_index_A_v3",
    "v3_dy": "/path/to/faiss_index_A_v3_dy",
    "v4": FAISS_INDEX_PATH,
}

# Retrieval defaults.
RETRIEVER_MAIN_TOPK = 3
RETRIEVER_SUB_TOPK = 3
RETRIEVER_MIN_SCORE = 0.9
STREAM_RETRIEVER_MIN_SCORE = 0.95
RERANK_TOP_N = 10
VECTOR_RETRIEVER_TOP_K = 50

# FastAPI service.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 50042
CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
MAX_CONVERSATION_TURNS = 5
MAX_CONVERSATION_HISTORY = 20

# Generation defaults for OpenAI-compatible chat completion APIs.
GENERATION_CONFIG_BASE = {
    "model": MODEL_NAME,
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 16384,
    "frequency_penalty": 0.05,
    "stop": None,
    "stream": True,
}

GENERATION_CONFIG_GREEDY = {
    "model": MODEL_NAME,
    "temperature": 0.0,
    "max_tokens": 16384,
    "stop": None,
    "stream": False,
}

RETRIEVER_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.8,
    "max_tokens": 16384,
    "frequency_penalty": 0.05,
    "stop": None,
    "stream": False,
}

TEMP_RESPONSE_TEMPERATURES = [0.1]
TEMP_RESPONSE_MAX_TOKENS = 8192
TEMP_RESPONSE_STREAM = True
ENABLE_THINKING = False
