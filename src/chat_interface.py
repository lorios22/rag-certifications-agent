import streamlit as st
import time
from datetime import datetime
from typing import List, Dict, Any
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_agent import create_certification_agent, RAGAgent
from config import CHAT_CONFIG, LOGS_DIR

logging.basicConfig(
    filename=LOGS_DIR / "chat_interface.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChatInterface:
    def __init__(self):
        self.agent = None
        self.conversation_history = []
        self.stats = {
            "total_questions": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }

    def initialize_agent(self):
        if self.agent is None:
            try:
                from config import PDF_PATH, VECTORSTORE_PATH
                import os
                
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                vectorstore_exists = os.path.exists(VECTORSTORE_PATH)
                
                if vectorstore_exists:
                    status_placeholder.info("Cargando vectorstore existente...")
                    progress_bar.progress(30)
                else:
                    status_placeholder.warning("Primera inicialización, procesando PDF...")
                    progress_bar.progress(10)
                
                self.agent = create_certification_agent(
                    pdf_path=PDF_PATH,
                    vectorstore_path=VECTORSTORE_PATH
                )
                
                progress_bar.progress(100)
                status_placeholder.success("Agente inicializado correctamente")
                logger.info("Agente RAG inicializado")
                
                time.sleep(2)
                status_placeholder.empty()
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"Error al inicializar: {e}")
                logger.error(f"Error: {e}")
                import traceback
                st.error(traceback.format_exc())
                return False
        return True

    def add_message(self, role: str, content: str, metadata: Dict = None):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.conversation_history.append(message)
        
        if len(self.conversation_history) > CHAT_CONFIG["max_history"]:
            self.conversation_history = self.conversation_history[-CHAT_CONFIG["max_history"]:]

    def ask_question(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            result = self.agent.ask(question)
            response_time = time.time() - start_time
            
            self.stats["total_questions"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["avg_response_time"] = self.stats["total_response_time"] / self.stats["total_questions"]
            
            result["metadata"]["response_time"] = response_time
            
            logger.info(f"Pregunta procesada en {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error al procesar pregunta: {e}")
            
            return {
                "success": False,
                "answer": "Error al procesar la pregunta.",
                "metadata": {}
            }

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        stats["conversation_length"] = len(self.conversation_history)
        
        if self.agent:
            agent_stats = self.agent.get_stats()
            stats["vectorstore_documents"] = agent_stats.get("vectorstore_documents", 0)
        
        return stats

    def clear_history(self):
        self.conversation_history = []
        self.stats = {
            "total_questions": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
        logger.info("Historial limpiado")


def display_message(message: Dict[str, Any]):
    role = message["role"]
    content = message["content"]
    metadata = message.get("metadata", {})
    
    with st.chat_message(role):
        st.write(content)
        
        if CHAT_CONFIG["show_sources"] and role == "assistant" and "sources" in metadata:
            sources = metadata.get("sources", [])
            if sources:
                with st.expander(f"Fuentes consultadas ({len(sources)})"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Fuente {i}:**")
                        st.text(source.get("content_preview", ""))
                        st.divider()


def create_sidebar(chat_interface: ChatInterface):
    with st.sidebar:
        st.header("Opciones")
        
        if st.button("Limpiar historial"):
            chat_interface.clear_history()
            st.rerun()
        
        st.divider()
        
        if CHAT_CONFIG["show_stats"]:
            st.subheader("Estadísticas")
            
            stats = chat_interface.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Preguntas", stats.get("total_questions", 0))
                st.metric("Tiempo promedio", f"{stats.get('avg_response_time', 0):.2f}s")
            with col2:
                st.metric("Longitud chat", stats.get("conversation_length", 0))
                if chat_interface.agent:
                    st.metric("Documentos", stats.get("vectorstore_documents", 0))
        
        st.divider()
        st.subheader("Información")
        
        if chat_interface.agent:
            st.write("Estado: Conectado")
            st.write(f"Modelo: {chat_interface.agent.llm_model}")
        else:
            st.write("Estado: Desconectado")
        
        st.divider()
        st.subheader("Instrucciones")
        st.write("Escribe tu pregunta sobre certificaciones AWS ML")


def main():
    st.set_page_config(
        page_title=CHAT_CONFIG["title"],
        page_icon="🤖",
        layout="wide"
    )
    
    st.title(CHAT_CONFIG["title"])
    st.markdown(CHAT_CONFIG["description"])
    
    if "chat_interface" not in st.session_state:
        st.session_state.chat_interface = ChatInterface()
    
    chat_interface = st.session_state.chat_interface
    
    create_sidebar(chat_interface)
    
    if not chat_interface.initialize_agent():
        st.error("No se pudo inicializar el agente")
        return
    
    chat_container = st.container()
    
    with chat_container:
        for message in chat_interface.conversation_history:
            display_message(message)
    
    if prompt := st.chat_input(CHAT_CONFIG["placeholder"]):
        chat_interface.add_message("user", prompt)
        
        with st.spinner("Procesando..."):
            result = chat_interface.ask_question(prompt)
        
        metadata = result.get("metadata", {})
        chat_interface.add_message("assistant", result["answer"], metadata)
        
        st.rerun()


if __name__ == "__main__":
    main()
