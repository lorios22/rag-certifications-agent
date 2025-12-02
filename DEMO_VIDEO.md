# Guía para Video de Demostración (10-12 minutos)

## PREPARACIÓN

```bash
cd rag-certifications-agent
source venv/bin/activate
python run_chat.py
```

Tener listos:
- Terminal con código
- Navegador en http://localhost:8501
- Editor con estructura del proyecto

---

## GUIÓN DEL VIDEO

### INTRODUCCIÓN (1 min)

**[Pantalla: Título]**

"Hola, este es un sistema RAG para consultas sobre certificaciones AWS Machine Learning.

El problema: empleados necesitan consultar información sin leer documentos extensos.

La solución: agente conversacional con respuestas precisas basadas en el documento oficial de AWS."

---

### ARQUITECTURA (2 min)

**[Pantalla: Estructura de carpetas]**

"Proyecto modular:

- data/: PDF y base de datos vectorial
- src/: Código fuente
  - rag_agent.py: Motor RAG
  - chat_interface.py: Interfaz Streamlit
  - config.py: Configuración
  - evaluator.py: Evaluación

Tecnologías:
- LangChain + OpenAI GPT-3.5
- Sentence Transformers para embeddings
- ChromaDB como base vectorial
- Streamlit para interfaz"

---

### FUNCIONAMIENTO (2 min)

**[Pantalla: Código rag_agent.py]**

"Proceso en 5 pasos:

1. Extrae texto del PDF
2. Divide en chunks de 1500 caracteres
3. Genera embeddings (vectores de 384 dimensiones)
4. Guarda en ChromaDB
5. Para responder:
   - Busca 3 chunks más relevantes
   - Combina como contexto
   - Envía a GPT-3.5"

---

### DEMO EN VIVO (5 min)

**[Pantalla: Interfaz Web]**

"Veamos el sistema en acción.

**Pregunta 1:**
[Escribir]: '¿Cuáles son los dominios del examen AWS ML?'

[Mostrar respuesta con 4 dominios y porcentajes]

Vean las fuentes consultadas abajo.

**Pregunta 2:**
[Escribir]: '¿Cuánto cuesta y cuánto dura el examen?'

[Mostrar]: 750 dólares, 180 minutos.

**Pregunta 3:**
[Escribir]: '¿Qué tipos de preguntas tiene el examen?'

[Mostrar respuesta detallada]

**[Sidebar]**
- Estadísticas de uso
- Documentos en vectorstore
- Tiempo de respuesta"

---

### INSTALACIÓN (1 min)

**[Pantalla: Terminal]**

"Para usar:

```bash
git clone https://github.com/lorios22/rag-certifications-agent.git
cd rag-certifications-agent
pip install -r requirements.txt
cp env.example .env
# Editar .env con OPENAI_API_KEY
python run_chat.py
```

Listo en http://localhost:8501"

---

### CONCLUSIÓN (1 min)

**[Pantalla: Resumen]**

"En resumen:

- Sistema RAG completo
- Respuestas precisas con fuentes
- Interfaz intuitiva
- Fácil instalación

Casos de uso:
- RR.HH: info de certificaciones
- Devs: preparación exámenes
- Gestores: planificación

Código en GitHub, completamente documentado.

Gracias."

---

## TIPS GRABACIÓN

### Técnico
- Resolución: 1920x1080
- Cerrar notificaciones
- Modo consistente (oscuro/claro)

### Grabación
- Hablar claro y pausado
- Hacer zoom en código importante
- Pausas entre secciones

### Edición
- Agregar títulos
- Resaltar código clave
- Timestamps en descripción

---

## CHECKLIST

- [ ] Sistema funcionando
- [ ] Navegador pantalla completa
- [ ] Terminal limpia
- [ ] Micrófono ok
- [ ] Notificaciones off

---

## PREGUNTAS PARA EL VIDEO

1. ¿Cuáles son los dominios del examen?
2. ¿Cuánto cuesta el examen?
3. ¿Cuánto tiempo dura?
4. ¿Qué tipos de preguntas tiene?

---

## DURACIÓN POR SECCIÓN

- Introducción: 1 min
- Arquitectura: 2 min
- Funcionamiento: 2 min
- Demo en vivo: 5 min
- Instalación: 1 min
- Conclusión: 1 min

**Total: 12 minutos**
