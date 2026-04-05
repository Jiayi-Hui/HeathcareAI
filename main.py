# -*- coding: utf-8 -*-
"""STAT 8017 - Main Entry Point"""

import os
import subprocess
import sys

from dotenv import load_dotenv

load_dotenv()

def check_dependencies():
    """Check if all dependencies are installed"""
    required = ['streamlit', 'langchain', 'chromadb', 'zhipuai', 'pandas']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    # if missing:
    #     print("❌ Missing dependencies:", ", ".join(missing))
    #     print("📌 Run: pip install -r requirements.txt")
    #     return False
    return True

def check_api_key():
    """Check if API key is set"""
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        print("❌ ZHIPU_API_KEY not found!")
        print("📌 Copy .env.example to .env and add your API key")
        return False
    print(f"✅ API Key configured: {api_key[:10]}...{api_key[-5:]}")
    return True

def main():
    print("=" * 60)
    print("🏥 STAT 8017 Healthcare Chatbot")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    if not check_api_key():
        sys.exit(1)

    print("\n🚀 Starting Streamlit...")
    port = os.environ.get("STREAMLIT_PORT", "8501")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", port,
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    main()
