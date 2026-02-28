"""
Markdown RAG MCP Server — Quick Install Script
Installs all dependencies and verifies the setup.
"""

import sys
import os
import subprocess
import platform

REQUIRED_PYTHON = (3, 10)
PACKAGES = [
    "chromadb",
    "sentence-transformers",
    "fastmcp",
    "rank-bm25",
]
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def check_python():
    version = sys.version_info
    if version < REQUIRED_PYTHON:
        print(f"ERROR: Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+ required.")
        print(f"       You have Python {version.major}.{version.minor}")
        sys.exit(1)
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro}")


def install_packages():
    print("\nInstalling Python packages...")
    for pkg in PACKAGES:
        print(f"  Installing {pkg}...", end=" ", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("OK")
        else:
            print(f"FAILED\n{result.stderr}")
            sys.exit(1)


def download_models():
    print("\nDownloading embedding models (one-time, ~520MB)...")
    print("  This may take 3-5 minutes on first run.")

    try:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
        from sentence_transformers import SentenceTransformer, CrossEncoder

        print(f"\n  [1/2] Downloading {EMBED_MODEL} (~120MB)...", flush=True)
        SentenceTransformer(EMBED_MODEL)
        print("        Done.")

        print(f"\n  [2/2] Downloading {RERANK_MODEL} (~80MB)...", flush=True)
        CrossEncoder(RERANK_MODEL)
        print("        Done.")

    except Exception as e:
        print(f"\n  WARNING: Could not pre-download models: {e}")
        print("  Models will download on first search instead.")


def show_config():
    server_path = os.path.abspath("server.py")
    python_path = sys.executable

    # Windows path formatting
    if platform.system() == "Windows":
        server_path = server_path.replace("\\", "\\\\")
        python_path = python_path.replace("\\", "\\\\")
        command = python_path
    else:
        command = "python"

    docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")).replace("\\", "\\\\")
    db_path = os.path.abspath("chroma_db").replace("\\", "\\\\")

    config = f'''{{
  "mcpServers": {{
    "markdown-rag": {{
      "command": "{command}",
      "args": ["{server_path}"],
      "env": {{
        "RAG_DOCS_PATH": "{docs_path}",
        "RAG_DB_PATH": "{db_path}"
      }}
    }}
  }}
}}'''

    print("\n" + "="*60)
    print("INSTALLATION COMPLETE!")
    print("="*60)
    print("\nAdd this to your MCP config file (mcp_config.json):")
    print("\n" + config)
    print("\nConfig file locations:")
    print("  Antigravity : C:\\Users\\<name>\\.gemini\\antigravity\\mcp_config.json")
    print("  Claude      : %APPDATA%\\Claude\\claude_desktop_config.json")
    print("  Cursor      : .cursor/mcp.json  (project root)")
    print("  Windsurf    : ~/.codeium/windsurf/mcp_config.json")
    print("\nAfter configuring, restart your IDE and run:")
    print("  > index_documents()   ← index your docs")
    print("  > search_docs('your question here')")
    print("="*60)


if __name__ == "__main__":
    print("Markdown RAG MCP Server — Installer")
    print("====================================\n")
    check_python()
    install_packages()
    download_models()
    show_config()
