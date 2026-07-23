"""
FlickMatrix AI — Unified Service Launcher

Launches both FastAPI REST API backend (port 8000) and Streamlit Frontend (port 8501)
simultaneously using background subprocesses.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Set working directory to project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

PYTHON_EXE = sys.executable

print("==================================================")
print("     🚀 FLICKMATRIX AI — LAUNCHING SERVICES      ")
print("==================================================")

# 1. Start FastAPI backend
print("\n[1/2] Starting FastAPI Backend Server (http://localhost:8000)...")
backend_cmd = [
    PYTHON_EXE,
    "-m",
    "uvicorn",
    "api.main:app",
    "--host",
    "127.0.0.1",
    "--port",
    "8000",
    "--reload",
]

backend_process = subprocess.Popen(backend_cmd, cwd=str(PROJECT_ROOT))
time.sleep(3)  # Allow API server to initialize model container

# 2. Start Streamlit frontend
print("\n[2/2] Starting Streamlit UI Dashboard (http://localhost:8501)...")
frontend_cmd = [
    PYTHON_EXE,
    "-m",
    "streamlit",
    "run",
    "frontend/app.py",
    "--server.port",
    "8501",
]

frontend_process = subprocess.Popen(frontend_cmd, cwd=str(PROJECT_ROOT))

print("\n==================================================")
print("  ✅ BOTH SERVICES ONLINE!")
print("  ----------------------------------------------")
print("  📌 FastAPI Backend Docs : http://localhost:8000/docs")
print("  📌 Streamlit Dashboard  : http://localhost:8501")
print("==================================================")
print("Press CTRL+C in this terminal to stop both servers.\n")

try:
    # Keep main script alive
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down FlickMatrix AI servers...")
    backend_process.terminate()
    frontend_process.terminate()
    print("Services stopped.")
