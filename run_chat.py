#!/usr/bin/env python3

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_requirements():
    try:
        import streamlit
        from dotenv import load_dotenv
        return True
    except ImportError as e:
        print(f"Error: {e}")
        print("Ejecuta: pip install -r requirements.txt")
        return False


def check_environment():
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY no configurada")
        print("Crea un archivo .env con tu API key")
        return False
    
    return True


def main():
    print("Iniciando interfaz de chat...")
    print("=" * 60)
    
    if not check_requirements():
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    chat_script = Path(__file__).parent / "src" / "chat_interface.py"
    
    if not chat_script.exists():
        print(f"Error: No se encuentra {chat_script}")
        sys.exit(1)
    
    print("Verificaciones completadas")
    print("Iniciando Streamlit...")
    print("App disponible en: http://localhost:8501")
    print("Presiona Ctrl+C para detener")
    print("=" * 60)
    
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", str(chat_script)]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nAplicación detenida")
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
