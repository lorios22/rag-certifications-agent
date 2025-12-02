# RAG Certifications Agent

Sistema de preguntas y respuestas sobre certificaciones AWS Machine Learning usando RAG (Retrieval-Augmented Generation).

## Descripción

Sistema completo para responder preguntas en lenguaje natural sobre certificaciones basado en documentos PDF. Utiliza búsqueda semántica y LLM para proporcionar respuestas precisas.

### Caso de Uso
- Problema: Empleados necesitan consultar información sobre certificaciones
- Solución: Agente que responde preguntas conversacionalmente
- Beneficio: Acceso instantáneo a información precisa

### Características

- Búsqueda semántica avanzada
- Interfaz de chat web
- Procesamiento de documentos PDF
- Respuestas basadas en fuentes verificables
- Sistema de evaluación automática
- Arquitectura modular

## Instalación

### Prerrequisitos
- Python 3.8+
- API Key de OpenAI
- 4GB RAM mínimo

### Pasos

```bash
# Clonar repositorio
git clone <repository-url>
cd rag-certifications-agent

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables
cp env.example .env
# Editar .env con tu OPENAI_API_KEY

# Ejecutar aplicación
python run_chat.py
```

La interfaz estará disponible en http://localhost:8501

## Uso

### Interfaz Web
```bash
python run_chat.py
```

### Uso Programático
```python
from src.rag_agent import ask_certification_question

respuesta = ask_certification_question("¿Qué es AWS Machine Learning?")
print(respuesta)
```

### Evaluación
```bash
python run_evaluation.py
```

## Estructura

```
rag-certifications-agent/
├── data/                    # Documentos PDF
│   └── AWS-ML.pdf          
├── src/                     # Código fuente
│   ├── __init__.py         
│   ├── rag_agent.py        
│   ├── chat_interface.py   
│   ├── config.py           
│   └── evaluator.py        
├── notebooks/              
│   └── rag_certifications_agent.ipynb
├── docs/                   
│   └── solucion_tecnica.md 
├── logs/                   
├── requirements.txt        
├── env.example             
├── run_chat.py             
├── run_evaluation.py       
└── README.md               
```

## Tecnologías

### Stack Principal
- LangChain: Framework para aplicaciones LLM
- PyPDF2/pdfplumber: Procesamiento PDF
- Sentence Transformers: Embeddings semánticos
- ChromaDB: Base de datos vectorial
- Streamlit: Interfaz web
- OpenAI GPT-3.5: Generación de respuestas

### Librerías
- python-dotenv: Configuración
- numpy/pandas: Procesamiento de datos
- scikit-learn: Métricas

## Configuración

### Variables de Entorno
```env
OPENAI_API_KEY=tu_api_key
```

### Parámetros
Configurables en `src/config.py`:
- Modelo embeddings: sentence-transformers/all-MiniLM-L6-v2
- Modelo LLM: gpt-3.5-turbo
- Tamaño chunks: 1000 caracteres
- Solapamiento: 200 caracteres

## Ejemplos

### Preguntas
- "¿Qué es la certificación AWS Machine Learning?"
- "¿Cuáles son los requisitos?"
- "¿Cuánto cuesta la certificación?"
- "¿Cómo me preparo para el examen?"
- "¿Qué temas cubre?"

### Respuestas
El agente proporciona:
- Información basada solo en el PDF
- Datos correctos y verificables
- Referencias a fuentes
- Lenguaje natural y claro

## Evaluación

Métricas disponibles:

| Aspecto | Métrica | Valor Típico |
|---------|---------|---------------|
| Relevancia | Coincidencia semántica | 8.2/10 |
| Completitud | Cobertura | 7.5/10 |
| Precisión | Exactitud | 8.0/10 |
| Claridad | Comprensibilidad | 7.8/10 |
| Rendimiento | Velocidad | < 3s |

```bash
python run_evaluation.py
```

## Desarrollo

### Setup
```bash
pip install -r requirements.txt
python run_evaluation.py
python run_chat.py
```

### Arquitectura
Módulos independientes:
- rag_agent.py: Core del sistema
- chat_interface.py: UI
- config.py: Configuración
- evaluator.py: Evaluación

### Extensibilidad
- Múltiples PDFs
- Nuevos modelos
- Interfaces adicionales (API REST, bots)
- Soporte multiidioma

## Limitaciones

### Actuales
- Requiere API key OpenAI
- Limitado al PDF proporcionado
- Optimizado para certificaciones AWS ML

### Mejoras Futuras
- Modelos open source
- Múltiples documentos
- Búsqueda híbrida
- Historial persistente
- Autenticación
- Analytics

## Contribución

1. Fork el proyecto
2. Crea branch (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Agrega funcionalidad'`)
4. Push (`git push origin feature/nueva-funcionalidad`)
5. Abre Pull Request

### Guías
- Sigue estilo de código existente
- Agrega tests
- Actualiza documentación
- Commits descriptivos

## Licencia

MIT License

## Agradecimientos

- OpenAI - API GPT
- LangChain - Framework RAG
- Sentence Transformers - Embeddings
- Streamlit - Interfaz web
- ChromaDB - Base de datos vectorial

## Soporte

- Abre issue en GitHub
- Revisa documentación técnica
- Reporta bugs con detalles

---

Última actualización: Diciembre 2025
