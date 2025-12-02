# Guía para Grabar Video de Demostración

## Script de Demostración

Este documento proporciona el guión completo para grabar un video explicando el sistema RAG de certificaciones AWS ML.

---

## PREPARACIÓN (Antes de Grabar)

### 1. Iniciar el Sistema
```bash
cd rag-certifications-agent
source venv/bin/activate
python run_chat.py
```

### 2. Abrir Navegador
- Abrir http://localhost:8501
- Tener listo el navegador a pantalla completa

### 3. Tener Listos
- Terminal con el código fuente abierto
- Editor con estructura de proyecto visible
- Navegador con la interfaz web

---

## GUIÓN DEL VIDEO (15-20 minutos)

### INTRODUCCIÓN (2 min)

**[Pantalla: Título del Proyecto]**

"Hola, voy a presentar un sistema RAG completo para consultas sobre certificaciones AWS Machine Learning.

Este proyecto resuelve un problema real: los empleados necesitan consultar información sobre certificaciones sin tener que leer documentos extensos.

La solución es un agente conversacional que responde preguntas precisas basadas en el documento oficial de AWS."

---

### PARTE 1: ARQUITECTURA DEL SISTEMA (3 min)

**[Pantalla: Estructura de carpetas]**

"El proyecto está organizado de forma modular:

- **data/**: Contiene el PDF de certificaciones AWS ML y la base de datos vectorial
- **src/**: Código fuente principal
  - rag_agent.py: Motor del sistema RAG
  - chat_interface.py: Interfaz web con Streamlit
  - config.py: Configuración centralizada
  - evaluator.py: Sistema de evaluación
- **run_chat.py**: Script para iniciar la aplicación
- **demo.py**: Script de demostración

**[Pantalla: Diagrama de flujo o código de rag_agent.py]**

La arquitectura utiliza:
- LangChain para orquestar el sistema RAG
- Sentence Transformers para embeddings semánticos
- ChromaDB como base de datos vectorial
- OpenAI GPT-3.5 para generación de respuestas
- Streamlit para la interfaz web"

---

### PARTE 2: CÓMO FUNCIONA EL SISTEMA (5 min)

**[Pantalla: Código de rag_agent.py - función initialize]**

"El sistema funciona en varios pasos:

**Paso 1: Procesamiento del PDF**
```python
# Extrae texto del PDF usando pdfplumber
pdf_text = self._extract_text_from_pdf(pdf_path)
```

**Paso 2: Creación de Chunks**
```python
# Divide el texto en chunks de 1500 caracteres con overlap de 300
# Esto asegura contexto completo sin cortar información importante
text_chunks = self._create_text_chunks(pdf_text)
```

**Paso 3: Generación de Embeddings**
- Convierte cada chunk en un vector numérico de 384 dimensiones
- Captura el significado semántico del texto
- Permite búsquedas por similitud, no por palabras exactas

**Paso 4: Almacenamiento Vectorial**
- Guarda los embeddings en ChromaDB
- Permite búsquedas rápidas y eficientes
- Se carga una sola vez y queda en memoria

**Paso 5: Respuesta a Preguntas**
```python
# 1. Usuario hace una pregunta
# 2. Sistema busca los 3 chunks más relevantes
source_documents = self.retriever.invoke(question)

# 3. Combina los chunks como contexto
context = "\n\n".join([doc.page_content for doc in source_documents])

# 4. Envía contexto + pregunta a GPT-3.5
answer = self.llm.invoke(formatted_prompt).content
```"

---

### PARTE 3: DEMOSTRACIÓN EN VIVO (7 min)

**[Pantalla: Interfaz Web de Streamlit]**

"Ahora veamos el sistema en acción.

**Pregunta 1: Estructura del Examen**

[Escribir en el chat]: '¿Cuáles son los 4 dominios del examen AWS Machine Learning?'

[Esperar respuesta]

Como pueden ver, el sistema responde con una lista clara:
- Dominio 1: Ingeniería de datos (20%)
- Dominio 2: Análisis exploratorio (24%)
- Dominio 3: Modelado (36%)
- Dominio 4: Implementación (20%)

Y aquí abajo podemos ver las fuentes consultadas. El sistema muestra exactamente qué partes del documento usó para responder.

**Pregunta 2: Información Específica**

[Escribir]: '¿Cuánto cuesta y cuánto dura el examen?'

[Esperar respuesta]

Respuesta directa: 750 dólares y 180 minutos. El sistema extrae datos específicos del documento.

**Pregunta 3: Detalles Técnicos**

[Escribir]: '¿Qué temas cubre el dominio de Modelado?'

[Esperar respuesta]

El sistema proporciona una respuesta detallada con los subtemas específicos.

**[Mostrar sidebar]**

En el panel lateral vemos:
- Estadísticas de uso
- Número de documentos en la base de datos
- Tiempo promedio de respuesta
- Opción para limpiar historial"

---

### PARTE 4: CÓDIGO Y CONFIGURACIÓN (3 min)

**[Pantalla: src/config.py]**

"El sistema es altamente configurable:

```python
# Parámetros de chunks
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

# Modelos utilizados
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'gpt-3.5-turbo'

# Número de fuentes a consultar
MAX_SOURCES = 3
```

**[Pantalla: src/rag_agent.py - función ask]**

El prompt está optimizado para respuestas directas:

```python
qa_template = '''
Eres un asistente experto en certificaciones AWS ML.
Responde DIRECTAMENTE usando SOLO la información del contexto.
Sé específico y concreto.
Si hay números o datos, menciónalos.
'''
```"

---

### PARTE 5: SCRIPT DE DEMOSTRACIÓN (2 min)

**[Pantalla: Terminal ejecutando demo.py]**

"Incluí un script de demostración completo:

```bash
python demo.py
```

[Ejecutar y mostrar salida]

Este script:
- Inicializa el sistema
- Hace preguntas de ejemplo
- Muestra las respuestas y fuentes
- Presenta estadísticas
- Explica casos de uso

Es perfecto para presentaciones o para entender rápidamente cómo funciona todo."

---

### PARTE 6: INSTALACIÓN Y USO (2 min)

**[Pantalla: README.md]**

"Para usar el sistema:

```bash
# 1. Clonar repositorio
git clone https://github.com/lorios22/rag-certifications-agent.git
cd rag-certifications-agent

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar API key
cp env.example .env
# Editar .env con tu OPENAI_API_KEY

# 5. Ejecutar
python run_chat.py
```

Y listo, la aplicación estará en http://localhost:8501"

---

### CONCLUSIÓN (1 min)

**[Pantalla: Resumen visual]**

"En resumen, este proyecto proporciona:

✓ Sistema RAG completo y funcional
✓ Respuestas precisas basadas en documentos oficiales
✓ Interfaz web intuitiva
✓ Código modular y extensible
✓ Fácil de instalar y usar

**Casos de uso:**
- Recursos Humanos: consultar información de certificaciones
- Desarrolladores: preparación para exámenes
- Gestores: planificación de programas de formación

**Tecnologías:**
- Python, LangChain, OpenAI GPT-3.5
- Sentence Transformers, ChromaDB
- Streamlit, pdfplumber

El código está disponible en GitHub y está completamente documentado.

Gracias por ver la demostración."

---

## TIPS PARA LA GRABACIÓN

### Preparación Técnica
- Resolución mínima: 1920x1080
- Usar modo oscuro/claro consistente en todo el video
- Cerrar notificaciones y aplicaciones innecesarias
- Tener buen micrófono

### Durante la Grabación
- Hablar claro y a buen ritmo
- Hacer pausas entre secciones
- Mostrar código importante en pantalla completa
- Hacer zoom en partes importantes del código

### Secciones que Puedes Grabar por Separado
1. Introducción y arquitectura
2. Explicación de código
3. Demostración en vivo
4. Instalación y conclusión

### Edición Post-Grabación
- Agregar títulos y subtítulos
- Resaltar líneas de código importantes
- Incluir diagramas si es posible
- Agregar música de fondo suave
- Incluir timestamps en la descripción

---

## CHECKLIST PRE-GRABACIÓN

- [ ] Sistema funcionando correctamente
- [ ] Navegador en pantalla completa
- [ ] Terminal limpia y lista
- [ ] Editor de código abierto con archivos correctos
- [ ] Micrófono funcionando
- [ ] Notificaciones desactivadas
- [ ] Aplicaciones innecesarias cerradas
- [ ] Guión revisado
- [ ] Ejemplos de preguntas preparadas

---

## PREGUNTAS DE EJEMPLO PARA EL VIDEO

1. ¿Cuáles son los dominios del examen?
2. ¿Cuánto cuesta el examen?
3. ¿Cuánto tiempo dura?
4. ¿Qué porcentaje corresponde a Modelado?
5. ¿Qué tipos de preguntas tiene el examen?
6. ¿Cuántas preguntas tiene el examen?
7. ¿Qué temas cubre el dominio de Ingeniería de datos?

---

## DURACIÓN SUGERIDA POR SECCIÓN

- Introducción: 2 min
- Arquitectura: 3 min
- Funcionamiento: 5 min
- Demostración en vivo: 7 min
- Código: 3 min
- Instalación: 2 min
- Conclusión: 1 min

**Total: 23 minutos**

Puedes ajustar según necesites. Para un video más corto (10-12 min), enfócate en la demostración en vivo.

