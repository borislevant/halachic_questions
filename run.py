"""Entry point for the Halachic Q&A application."""

import subprocess
import sys
from pathlib import Path

from src.config import load_config
from src.storage.database import initialize_database


def main() -> None:
    """Initialize the application and launch the Streamlit UI."""
    config = load_config()

    # Ensure required directories exist
    Path(config.storage.books_dir).mkdir(parents=True, exist_ok=True)
    Path(config.storage.processed_dir).mkdir(parents=True, exist_ok=True)
    Path(config.storage.chroma_dir).mkdir(parents=True, exist_ok=True)

    # Initialize SQLite database
    initialize_database(config.storage.sqlite_path)

    # Launch Streamlit
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/ui/app.py",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
        ],
        check=False,
    )


if __name__ == "__main__":
    main()
