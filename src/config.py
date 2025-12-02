import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
LOGS_DIR = PROJECT_ROOT / "logs"

PDF_PATH = str(DATA_DIR / "AWS-ML.pdf")
ENV_FILE = str(PROJECT_ROOT / ".env")
VECTORSTORE_PATH = str(VECTORSTORE_DIR)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MAX_SOURCES = 3

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

STREAMLIT_PORT = 8501
STREAMLIT_HOST = "localhost"

DATA_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

REQUIRED_ENV_VARS = ["OPENAI_API_KEY"]


def validate_environment():
    missing_vars = []
    
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(
            f"Faltan variables de entorno: {', '.join(missing_vars)}"
        )


def get_openai_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY no configurada")
    return api_key


AVAILABLE_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
]

AVAILABLE_LLM_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview"
]

QA_PROMPT_TEMPLATE = """
Eres un asistente especializado en certificaciones de AWS Machine Learning.
Responde basándote únicamente en la información del contexto.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
- Responde clara y concisamente
- Si no está en el contexto, indícalo
- Tono profesional y educativo
- Estructura lógica

Respuesta:
"""

EVALUATION_QUESTIONS = [
    "¿Qué es la certificación AWS Machine Learning?",
    "¿Cuáles son los requisitos para obtener la certificación?",
    "¿Cuánto cuesta la certificación AWS ML?",
    "¿Qué temas cubre el examen?",
    "¿Cómo prepararme para la certificación?",
    "¿Cuánto tiempo toma completar la certificación?",
    "¿Qué nivel de experiencia se requiere?",
    "¿Hay prerrequisitos para el examen?"
]

CHAT_CONFIG = {
    "title": "Asistente Certificaciones AWS ML",
    "description": "Pregunta sobre certificaciones de AWS Machine Learning",
    "placeholder": "Escribe tu pregunta...",
    "max_history": 50,
    "show_sources": True,
    "show_stats": True
}
