import sys
import subprocess

def main():
    print("─" * 60)
    print(f"Target Python environment: {sys.executable}")
    print("─" * 60)
    print("Installing PDF pipeline and RAG dependencies...")

    # Core libraries required by BetterIngest, captioning, and the backend server
    packages = [
        "paddlepaddle",
        "paddlex",
        "paddleocr",
        "pypdfium2",
        "pillow",
        "ollama",
        "fastapi",
        "uvicorn",
        "pydantic",
        "openai"
    ]

    try:
        # Upgrade pip first to avoid package resolution issues
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install the dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("\n✓ All dependencies successfully installed!")
    except subprocess.CalledProcessError as exc:
        print(f"\n✗ Error occurred during installation: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
