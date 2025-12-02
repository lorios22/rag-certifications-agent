# Script Detallado para Video - Sistema RAG Certificaciones AWS ML

## DURACIÓN TOTAL: 12 minutos

---

## [0:00 - 1:00] INTRODUCCIÓN

### PANTALLA: Pantalla en negro con título
```
SISTEMA RAG PARA CERTIFICACIONES AWS ML
Consultas Inteligentes con IA
```

### QUÉ DECIR:
"Hola, en este video voy a mostrar un sistema completo de RAG, que significa Retrieval-Augmented Generation, para hacer consultas sobre certificaciones de AWS Machine Learning.

El problema que resuelve es simple: los empleados necesitan consultar información específica sobre certificaciones sin tener que leer documentos PDF de 50 o 100 páginas.

La solución es un agente conversacional que usa inteligencia artificial para responder preguntas precisas basándose únicamente en el documento oficial de AWS."

---

## [1:00 - 3:00] ARQUITECTURA Y ESTRUCTURA

### PANTALLA 1: Terminal mostrando estructura de carpetas

### QUÉ HACER:
```bash
cd rag-certifications-agent
tree -L 2 -I 'venv|__pycache__|*.pyc'
```

### QUÉ DECIR:
"Primero veamos cómo está organizado el proyecto. Es una arquitectura modular muy clara:

- La carpeta **data** contiene el PDF de certificaciones AWS ML y la base de datos vectorial donde se guardan los embeddings.

- La carpeta **src** tiene todo el código fuente:
  - rag_agent.py es el motor principal del sistema RAG
  - chat_interface.py es la interfaz web con Streamlit
  - config.py tiene toda la configuración centralizada
  - evaluator.py permite evaluar la calidad del sistema

- run_chat.py es simplemente el script que inicia la aplicación

- demo.py es un script de demostración que voy a ejecutar después."

### PANTALLA 2: Abrir VSCode mostrando la estructura

### QUÉ DECIR:
"Las tecnologías principales son:
- LangChain para orquestar todo el sistema RAG
- Sentence Transformers para crear embeddings semánticos
- ChromaDB como base de datos vectorial
- OpenAI GPT-3.5 para generar las respuestas
- Streamlit para la interfaz web

Ahora veamos el código."

---

## [3:00 - 5:30] CONFIGURACIÓN Y PARÁMETROS

### ARCHIVO: src/config.py
### LÍNEAS A MOSTRAR: 1-35

### QUÉ HACER:
1. Abrir `src/config.py`
2. Hacer scroll hasta las líneas de configuración principales

### QUÉ DECIR:
"Empecemos por la configuración. Este archivo define todos los parámetros del sistema."

### RESALTAR Y EXPLICAR (líneas 23-33):
```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 500

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MAX_SOURCES = 3
```

### QUÉ DECIR:
"Los parámetros más importantes son:

- **CHUNK_SIZE** de 1500 caracteres: es el tamaño de cada fragmento de texto. Lo hice más grande que lo típico para mantener mejor contexto.

- **CHUNK_OVERLAP** de 300: significa que cada chunk se solapa 300 caracteres con el anterior para no perder información en los cortes.

- **MAX_SOURCES** 3: el sistema consulta los 3 fragmentos más relevantes para responder cada pregunta.

- El modelo de embeddings es **all-MiniLM-L6-v2** que genera vectores de 384 dimensiones.

- Y usamos **GPT-3.5-turbo** con temperatura baja de 0.1 para respuestas más precisas y consistentes."

---

## [5:30 - 8:00] CÓDIGO PRINCIPAL DEL RAG

### ARCHIVO: src/rag_agent.py
### MOSTRAR 4 FUNCIONES CLAVE

---

### FUNCIÓN 1: Extracción de PDF (30 seg)

### LÍNEAS: 70-85

```python
def _extract_text_from_pdf(self, pdf_path: str) -> str:
    logger.info(f"Extrayendo texto del PDF: {pdf_path}")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        logger.info(f"Texto extraído. Longitud: {len(text)} caracteres")
        return text
```

### QUÉ DECIR:
"Veamos cómo funciona el sistema paso a paso.

Primero, la función **_extract_text_from_pdf** usa pdfplumber para extraer todo el texto del PDF. Simplemente itera página por página y concatena todo el texto. Del documento de AWS ML extrae unos 14 mil caracteres."

---

### FUNCIÓN 2: Creación de Chunks (1 min)

### LÍNEAS: 87-108 (mostrar versión simplificada)

```python
def _create_text_chunks(self, text: str) -> List[str]:
    logger.info(f"Creando chunks (tamaño: {self.chunk_size}, overlap: {self.chunk_overlap})")
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + self.chunk_size
        
        # Buscar punto natural para cortar
        if end < len(text):
            for char in ['. ', '? ', '! ']:
                last_pos = text.rfind(char, start, end)
                if last_pos != -1 and last_pos > start:
                    end = last_pos + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Avanzar con overlap
        start = end - self.chunk_overlap
    
    logger.info(f"Creados {len(chunks)} chunks")
    return chunks
```

### QUÉ DECIR:
"Segundo paso: **_create_text_chunks** divide el texto en fragmentos.

Lo importante aquí es que no corta arbitrariamente en el carácter 1500. El algoritmo busca un punto natural - un punto, signo de interrogación o exclamación - para hacer el corte y no partir frases a la mitad.

Y como dije antes, cada chunk se solapa 300 caracteres con el anterior para no perder contexto. Con este documento genera 14 chunks."

---

### FUNCIÓN 3: Inicialización de Embeddings (30 seg)

### LÍNEAS: 110-118

```python
def _initialize_embeddings(self):
    logger.info(f"Inicializando embeddings: {self.embedding_model_name}")
    
    try:
        self.embedding_model = SentenceTransformerEmbeddings(
            model_name=self.embedding_model_name
        )
        logger.info("Modelo de embeddings cargado")
```

### QUÉ DECIR:
"Tercer paso: inicializar el modelo de embeddings.

Sentence Transformers convierte cada chunk de texto en un vector numérico de 384 dimensiones. Estos vectores capturan el significado semántico del texto, lo que permite buscar por similitud de conceptos, no solo por palabras exactas."

---

### FUNCIÓN 4: Responder Preguntas (1.5 min)

### LÍNEAS: 206-250 (mostrar partes clave)

```python
def ask(self, question: str) -> Dict[str, Any]:
    try:
        logger.info(f"Procesando pregunta: {question}")
        
        # 1. Buscar documentos relevantes
        source_documents = self.retriever.invoke(question)
        
        # 2. Combinar como contexto
        context = "\n\n".join([doc.page_content for doc in source_documents])
        
        # 3. Generar respuesta con GPT-3.5
        formatted_prompt = self.prompt.format(context=context, question=question)
        answer = self.llm.invoke(formatted_prompt).content
        
        # 4. Preparar metadata
        metadata = {
            "question": question,
            "num_sources": len(source_documents),
            "sources": [
                {
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "content_preview": doc.page_content[:400],
                    "source": doc.metadata.get("source")
                }
                for doc in source_documents
            ]
        }
        
        return {
            "success": True,
            "answer": answer,
            "metadata": metadata
        }
```

### QUÉ DECIR:
"Y aquí está la función más importante: **ask**, que responde preguntas.

El proceso tiene 4 pasos:

**Paso 1**: Cuando el usuario hace una pregunta, el sistema la convierte en un vector y busca en ChromaDB los 3 chunks más similares semánticamente usando `retriever.invoke`.

**Paso 2**: Combina esos 3 chunks en un solo contexto.

**Paso 3**: Envía ese contexto junto con la pregunta a GPT-3.5. El prompt está diseñado para que responda SOLO basándose en el contexto, sin inventar información.

**Paso 4**: Devuelve la respuesta junto con los metadatos: qué chunks consultó, de qué parte del documento vienen, etc.

Este enfoque garantiza que las respuestas sean verificables y basadas en el documento real."

---

## [8:00 - 8:30] PROMPT OPTIMIZADO

### ARCHIVO: src/rag_agent.py
### LÍNEAS: 176-191

```python
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
```

### QUÉ DECIR:
"Rápidamente les muestro el prompt optimizado.

Las instrucciones clave son:
- Responder DIRECTAMENTE usando solo el contexto
- Ser específico con números y datos
- Usar bullet points para claridad

Esto hace que las respuestas sean concisas y bien estructuradas."

---

## [8:30 - 13:00] DEMOSTRACIÓN EN VIVO

### PANTALLA: Navegador con http://localhost:8501

### QUÉ HACER ANTES:
```bash
python run_chat.py
# Esperar a que abra en el navegador
```

### QUÉ DECIR:
"Ahora veamos el sistema en acción. Ya tengo la aplicación corriendo en el navegador."

---

### PREGUNTA 1 (1.5 min)

### QUÉ ESCRIBIR:
```
¿Cuáles son los 4 dominios del examen AWS Machine Learning?
```

### QUÉ DECIR MIENTRAS ESCRIBE:
"Voy a hacerle una pregunta sobre la estructura del examen."

### DESPUÉS DE VER LA RESPUESTA:
"Perfecto. El sistema responde con una lista clara de los 4 dominios con sus porcentajes exactos:
- Ingeniería de datos: 20%
- Análisis exploratorio: 24%
- Modelado: 36%
- Implementación: 20%

Y aquí abajo vemos las fuentes consultadas. Si expandimos podemos ver exactamente qué fragmentos del PDF usó para responder. Esto hace que la respuesta sea completamente verificable."

### ACCIÓN: Expandir una fuente y mostrarla

---

### PREGUNTA 2 (1 min)

### QUÉ ESCRIBIR:
```
¿Cuánto cuesta el examen y cuánto tiempo tengo para completarlo?
```

### QUÉ DECIR:
"Segunda pregunta, ahora sobre detalles prácticos."

### DESPUÉS DE LA RESPUESTA:
"Respuesta directa: 750 dólares y 180 minutos. El sistema extrae los datos específicos del documento sin agregar información extra."

---

### PREGUNTA 3 (1.5 min)

### QUÉ ESCRIBIR:
```
¿Qué tipos de preguntas tiene el examen?
```

### DESPUÉS DE LA RESPUESTA:
"Aquí vemos una respuesta más detallada con bullet points:
- Preguntas de selección múltiple
- Cinco o más opciones
- Sin penalización por adivinar
- 50 preguntas con puntaje
- 15 preguntas de prueba sin puntaje

Todo estructurado y fácil de leer."

---

### MOSTRAR SIDEBAR (30 seg)

### QUÉ HACER:
Hacer clic en el sidebar y mostrar estadísticas

### QUÉ DECIR:
"En el panel lateral vemos estadísticas en tiempo real:
- Número de preguntas realizadas
- Tiempo promedio de respuesta
- Cantidad de documentos en la base vectorial
- Y podemos limpiar el historial si queremos empezar una conversación nueva.

El sistema mantiene el historial de la conversación durante la sesión."

---

## [13:00 - 14:00] INSTALACIÓN

### PANTALLA: Terminal nueva

### QUÉ HACER:
Mostrar el README o escribir los comandos en pantalla

### QUÉ DECIR:
"Para usar este sistema en tu propia computadora es muy simple.

Primero clonas el repositorio de GitHub:"

```bash
git clone https://github.com/lorios22/rag-certifications-agent.git
cd rag-certifications-agent
```

"Instalas las dependencias:"

```bash
pip install -r requirements.txt
```

"Configuras tu API key de OpenAI en el archivo .env:"

```bash
cp env.example .env
# Editas el .env con tu OPENAI_API_KEY
```

"Y ejecutas:"

```bash
python run_chat.py
```

"La aplicación abre automáticamente en localhost:8501 y ya puedes empezar a hacer preguntas."

---

## [14:00 - 15:00] CONCLUSIÓN

### PANTALLA: Volver al código o mostrar resumen visual

### QUÉ DECIR:
"Para resumir, este proyecto proporciona:

- Un sistema RAG completo y funcional
- Respuestas precisas basadas en fuentes verificables del documento oficial
- Una interfaz web intuitiva con Streamlit
- Código modular y fácil de extender

Los casos de uso principales son:

- Recursos Humanos puede consultar rápidamente información sobre certificaciones
- Desarrolladores pueden usarlo para prepararse para los exámenes
- Gestores pueden planificar programas de certificación con datos precisos

Todo el código está en GitHub, está completamente documentado, y es fácil de adaptar para otros documentos PDF.

Si tienen preguntas déjenlas en los comentarios. Gracias por ver el video."

### PANTALLA FINAL:
```
GitHub: github.com/lorios22/rag-certifications-agent

Tecnologías:
• Python • LangChain • OpenAI
• Sentence Transformers • ChromaDB
• Streamlit
```

---

## RESUMEN DE ARCHIVOS MOSTRADOS

| Tiempo | Archivo | Líneas | Qué Mostrar |
|--------|---------|--------|-------------|
| 3:00-5:30 | `src/config.py` | 23-33 | Parámetros principales |
| 5:30-6:00 | `src/rag_agent.py` | 70-85 | Extracción PDF |
| 6:00-7:00 | `src/rag_agent.py` | 87-108 | Creación chunks |
| 7:00-7:30 | `src/rag_agent.py` | 110-118 | Embeddings |
| 7:30-8:00 | `src/rag_agent.py` | 206-250 | Función ask |
| 8:00-8:30 | `src/rag_agent.py` | 176-191 | Prompt template |
| 8:30-13:00 | Navegador | - | Demo en vivo |

---

## TIPS FINALES

### Antes de Grabar:
- [ ] Cerrar todas las notificaciones
- [ ] Tener solo las pestañas necesarias abiertas
- [ ] Aumentar tamaño de fuente en VSCode (16-18pt)
- [ ] Terminal con fuente grande
- [ ] Probar micrófono

### Durante la Grabación:
- Habla pausado y claro
- Haz pausas de 2 segundos entre secciones
- Si te equivocas, pausa y reinicia esa sección
- No digas "um", "eh", etc. - pausa en silencio si necesitas pensar

### Edición:
- Cortar pausas largas
- Agregar títulos en cada sección
- Resaltar líneas de código importantes con zoom o recuadro
- Agregar timestamps en la descripción del video

