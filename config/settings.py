import os
from pathlib import Path

class Settings:
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_STORE_DIR = DATA_DIR / "vectorstore"
    
    # API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Model Configuration
    MODEL_NAME = "llama-3.1-70b-versatile"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    
    # Processing Configuration
    BATCH_SIZE = 32
    MAX_WORKERS = 4
    CHUNK_SIZE = 512
    OVERLAP_SIZE = 50
    
    
    # Logging Configuration
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "formatter": "default",
                "filename": str(BASE_DIR / "app.log"),
                "mode": "a"
            }
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"]
        },
        "loggers": {
            "app": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            }
        }
    }

    # Vector Store Configuration
    VECTOR_STORE_SETTINGS = {
        "distance_metric": "cosine",
        "persist_directory": str(VECTOR_STORE_DIR)
    }

    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)