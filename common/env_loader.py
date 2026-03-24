from pathlib import Path
from dotenv import load_dotenv


def load_env() -> Path:
    """Load environment variables from the project root .env file."""
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    load_dotenv(env_path)
    return env_path
