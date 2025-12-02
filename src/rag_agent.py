import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import pdfplumber
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import numpy as np
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()


class RAGAgent:
    def __init__(
        self,
        pdf_path: str,
        vectorstore_path: str = "./data/vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
        temperature: float = 0.1,
        openai_api_key: Optional[str] = None
    ):
        self.pdf_path = pdf_path
        self.vectorstore_path = vectorstore_path
        self.embedding_model_name = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        self.embedding_model = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.stats = {
            "pdf_processed": False,
            "chunks_created": 0,
            "vectorstore_loaded": False,
            "total_questions_answered": 0
        }
        
        logger.info(f"Inicializado agente RAG para PDF: {pdf_path}")
        logger.info(f"Modelo embeddings: {embedding_model}, LLM: {llm_model}")

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        logger.info(f"Extrayendo texto del PDF: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            logger.info(f"Texto extraído. Longitud: {len(text)} caracteres")
            return text
        except Exception as e:
            logger.error(f"Error al extraer texto del PDF: {e}")
            raise

    def _create_text_chunks(self, text: str) -> List[str]:
        if not text:
            return []
        
        logger.info(f"Creando chunks (tamaño: {self.chunk_size}, overlap: {self.chunk_overlap})")
        
        chunks = []
        start = 0
        iterations = 0
        max_iterations = 10000
        
        while start < len(text) and iterations < max_iterations:
            iterations += 1
            end = start + self.chunk_size
            
            if end < len(text):
                for char in ['. ', '? ', '! ']:
                    last_pos = text.rfind(char, start, end)
                    if last_pos != -1 and last_pos > start:
                        end = last_pos + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + max(1, self.chunk_size - self.chunk_overlap)
            start = next_start
        
        logger.info(f"Creados {len(chunks)} chunks")
        return chunks

    def _initialize_embeddings(self):
        logger.info(f"Inicializando embeddings: {self.embedding_model_name}")
        
        try:
            self.embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model_name)
            logger.info("Modelo de embeddings cargado")
        except Exception as e:
            logger.error(f"Error al cargar embeddings: {e}")
            raise

    def _initialize_vectorstore(self, documents: List[Document]):
        try:
            if len(documents) == 0:
                logger.info(f"Cargando vectorstore desde: {self.vectorstore_path}")
                self.vectorstore = Chroma(
                    persist_directory=self.vectorstore_path,
                    embedding_function=self.embedding_model
                )
                logger.info("Vectorstore cargado")
            else:
                logger.info(f"Creando vectorstore en: {self.vectorstore_path}")
                logger.info(f"Procesando {len(documents)} documentos en lotes")
                os.makedirs(self.vectorstore_path, exist_ok=True)
                
                batch_size = 5
                total_docs = len(documents)
                
                for i in range(0, total_docs, batch_size):
                    batch = documents[i:i+batch_size]
                    logger.info(f"Procesando lote {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
                    
                    if i == 0:
                        self.vectorstore = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embedding_model,
                            persist_directory=self.vectorstore_path
                        )
                    else:
                        self.vectorstore.add_documents(batch)
                
                logger.info(f"Vectorstore creado con {total_docs} documentos")
            
            try:
                if hasattr(self.vectorstore, 'persist'):
                    self.vectorstore.persist()
            except:
                pass
            
            self.stats["vectorstore_loaded"] = True
            
            doc_count = self.vectorstore._collection.count() if hasattr(self.vectorstore, '_collection') else len(documents)
            logger.info(f"Vectorstore inicializado con {doc_count} documentos")
            
        except Exception as e:
            logger.error(f"Error al inicializar vectorstore: {e}")
            raise

    def _initialize_qa_chain(self):
        logger.info(f"Inicializando chain QA con modelo: {self.llm_model}")
        
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            max_tokens=500
        )
        
        qa_template = """Eres un asistente experto en certificaciones AWS Machine Learning.

Contexto:
{context}

Pregunta: {question}

Instrucciones:
- Responde DIRECTAMENTE la pregunta usando SOLO la información del contexto
- Si la información no está en el contexto, di claramente "No encuentro esa información en el documento"
- Sé específico y concreto, evita rodeos
- Si hay números, fechas o datos específicos, menciónalos
- Usa bullet points cuando sea apropiado para claridad

Respuesta:"""
        
        self.prompt = PromptTemplate.from_template(qa_template)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        logger.info("Chain QA inicializado")

    def initialize(self):
        logger.info("Iniciando configuración del agente RAG")
        
        try:
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF no encontrado: {self.pdf_path}")
            
            self._initialize_embeddings()
            
            vectorstore_db_file = os.path.join(self.vectorstore_path, "chroma.sqlite3")
            need_to_create_vectorstore = True
            
            if os.path.exists(vectorstore_db_file):
                logger.info("Vectorstore existente encontrado")
                try:
                    self._initialize_vectorstore([])
                    doc_count = self.vectorstore._collection.count() if hasattr(self.vectorstore, '_collection') else 0
                    if doc_count == 0:
                        raise ValueError("Vectorstore vacío")
                    logger.info(f"Vectorstore cargado con {doc_count} documentos")
                    need_to_create_vectorstore = False
                except Exception as e:
                    logger.warning(f"Error al cargar vectorstore: {e}. Recreando")
                    import shutil
                    if os.path.exists(self.vectorstore_path):
                        shutil.rmtree(self.vectorstore_path)
                    need_to_create_vectorstore = True
            else:
                logger.info("No se encontró vectorstore existente")
            
            if need_to_create_vectorstore:
                logger.info("Procesando PDF")
                pdf_text = self._extract_text_from_pdf(self.pdf_path)
                
                logger.info("Creando chunks")
                text_chunks = self._create_text_chunks(pdf_text)
                self.stats["chunks_created"] = len(text_chunks)
                logger.info(f"Chunks creados: {len(text_chunks)}")
                
                logger.info("Creando documentos")
                self.documents = [
                    Document(page_content=chunk, metadata={"source": self.pdf_path, "chunk_id": i})
                    for i, chunk in enumerate(text_chunks)
                ]
                logger.info(f"{len(self.documents)} documentos creados")
                
                logger.info("Creando vectorstore")
                self._initialize_vectorstore(self.documents)
                self.stats["pdf_processed"] = True
                logger.info("Vectorstore creado")
            
            self._initialize_qa_chain()
            
            logger.info("Agente RAG inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error durante la inicialización: {e}")
            raise

    def ask(self, question: str) -> Dict[str, Any]:
        if not self.retriever:
            raise RuntimeError("Agente no inicializado. Llama a initialize() primero")
        
        try:
            logger.info(f"Procesando pregunta: {question}")
            
            source_documents = self.retriever.invoke(question)
            context = "\n\n".join([doc.page_content for doc in source_documents])
            
            formatted_prompt = self.prompt.format(context=context, question=question)
            answer = self.llm.invoke(formatted_prompt).content
            
            metadata = {
                "question": question,
                "num_sources": len(source_documents),
                "sources": [
                    {
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "content_preview": doc.page_content[:400] if len(doc.page_content) > 400 else doc.page_content,
                        "source": doc.metadata.get("source")
                    }
                    for doc in source_documents
                ],
                "timestamp": str(np.datetime64('now'))
            }
            
            self.stats["total_questions_answered"] += 1
            logger.info(f"Respuesta generada. Fuentes consultadas: {len(source_documents)}")
            
            return {
                "success": True,
                "answer": answer,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error al procesar pregunta: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": str(e),
                "answer": "Lo siento, ocurrió un error al procesar tu pregunta.",
                "metadata": {}
            }

    def get_stats(self) -> Dict[str, Any]:
        stats = self.stats.copy()
        
        if self.vectorstore:
            try:
                stats["vectorstore_documents"] = self.vectorstore._collection.count() if hasattr(self.vectorstore, '_collection') else 0
            except:
                stats["vectorstore_documents"] = 0
        
        return stats

    def search_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            raise RuntimeError("Agente no inicializado")
        
        similar_docs = self.vectorstore.similarity_search(query, k=k)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "chunk_id": doc.metadata.get("chunk_id")
            }
            for doc in similar_docs
        ]


def create_certification_agent(pdf_path: str = None, vectorstore_path: str = None) -> RAGAgent:
    if pdf_path is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        pdf_path = str(project_root / "data" / "AWS-ML.pdf")
    
    if vectorstore_path is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        vectorstore_path = str(project_root / "data" / "vectorstore")
    
    agent = RAGAgent(
        pdf_path=pdf_path,
        vectorstore_path=vectorstore_path
    )
    
    agent.initialize()
    
    return agent


def ask_certification_question(question: str, agent: Optional[RAGAgent] = None) -> str:
    if agent is None:
        agent = create_certification_agent()
    
    result = agent.ask(question)
    return result["answer"]
