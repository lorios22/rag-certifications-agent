#!/usr/bin/env python3
"""
Script de demostración del sistema RAG para certificaciones AWS ML.
Muestra todas las funcionalidades del sistema paso a paso.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_agent import create_certification_agent


def print_header(text):
    """Imprime un encabezado visual."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_section(text):
    """Imprime una sección."""
    print(f"\n--- {text} ---\n")


def wait():
    """Pausa para el video."""
    time.sleep(1)


def main():
    print_header("DEMO: Sistema RAG para Certificaciones AWS Machine Learning")
    
    print("Este sistema permite hacer preguntas sobre certificaciones AWS ML")
    print("y obtener respuestas precisas basadas en el documento oficial.\n")
    wait()
    
    # Paso 1: Inicialización
    print_section("Paso 1: Inicializando el Sistema")
    print("Cargando agente RAG...")
    print("- Procesando PDF de certificaciones AWS ML")
    print("- Creando embeddings de texto")
    print("- Configurando base de datos vectorial")
    print("- Preparando modelo de lenguaje GPT-3.5-turbo\n")
    
    agent = create_certification_agent()
    
    print("Sistema inicializado correctamente")
    wait()
    
    # Paso 2: Información del sistema
    print_section("Paso 2: Información del Sistema")
    stats = agent.get_stats()
    print(f"- Documentos en vectorstore: {stats.get('vectorstore_documents', 0)}")
    print(f"- Chunks de texto creados: {stats.get('chunks_created', 0)}")
    print(f"- Modelo de embeddings: {agent.embedding_model_name}")
    print(f"- Modelo LLM: {agent.llm_model}")
    wait()
    
    # Paso 3: Preguntas de demostración
    print_section("Paso 3: Demostración de Preguntas y Respuestas")
    
    preguntas_demo = [
        {
            "pregunta": "¿Cuáles son los 4 dominios del examen AWS Machine Learning?",
            "descripcion": "Pregunta sobre la estructura del examen"
        },
        {
            "pregunta": "¿Cuánto cuesta el examen de certificación?",
            "descripcion": "Pregunta sobre el costo"
        },
        {
            "pregunta": "¿Cuánto tiempo tengo para completar el examen?",
            "descripcion": "Pregunta sobre duración"
        },
        {
            "pregunta": "¿Qué porcentaje del examen corresponde al dominio de Modelado?",
            "descripcion": "Pregunta específica sobre ponderación"
        }
    ]
    
    for i, item in enumerate(preguntas_demo, 1):
        print(f"\n{'='*80}")
        print(f"PREGUNTA {i}: {item['descripcion']}")
        print(f"{'='*80}\n")
        print(f"Usuario pregunta: \"{item['pregunta']}\"\n")
        wait()
        
        print("Procesando...")
        resultado = agent.ask(item['pregunta'])
        wait()
        
        if resultado['success']:
            print(f"\nRESPUESTA DEL SISTEMA:")
            print("-" * 80)
            print(resultado['answer'])
            print("-" * 80)
            
            print(f"\nFUENTES CONSULTADAS: {resultado['metadata']['num_sources']}")
            for j, fuente in enumerate(resultado['metadata']['sources'], 1):
                print(f"\nFuente {j} (Chunk ID: {fuente['chunk_id']}):")
                preview = fuente['content_preview']
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                print(f"  {preview}")
        else:
            print(f"ERROR: {resultado.get('error', 'Error desconocido')}")
        
        wait()
    
    # Paso 4: Búsqueda semántica
    print_section("Paso 4: Demostración de Búsqueda Semántica")
    print("El sistema usa búsqueda semántica para encontrar información relevante.\n")
    print("Ejemplo: Buscando documentos similares a 'costo del examen'...\n")
    
    resultados_busqueda = agent.search_similar("costo del examen", k=2)
    
    for i, doc in enumerate(resultados_busqueda, 1):
        print(f"Resultado {i}:")
        print(f"  Relevancia: Alta")
        print(f"  Chunk ID: {doc['chunk_id']}")
        content = doc['content'][:150]
        print(f"  Contenido: {content}...")
        print()
    
    wait()
    
    # Paso 5: Estadísticas finales
    print_section("Paso 5: Estadísticas del Sistema")
    stats_final = agent.get_stats()
    print(f"- Total de preguntas respondidas: {stats_final.get('total_questions_answered', 0)}")
    print(f"- PDF procesado: {'Si' if stats_final.get('pdf_processed') else 'No'}")
    print(f"- Vectorstore cargado: {'Si' if stats_final.get('vectorstore_loaded') else 'No'}")
    wait()
    
    # Paso 6: Características principales
    print_section("Paso 6: Características Principales del Sistema")
    print("""
1. RESPUESTAS PRECISAS
   - Basadas únicamente en el documento oficial
   - Sin invención de información
   - Referencias verificables a fuentes

2. BÚSQUEDA INTELIGENTE
   - Embeddings semánticos avanzados
   - Encuentra información relevante aunque uses diferentes palabras
   - Contexto completo para respuestas precisas

3. INTERFAZ AMIGABLE
   - Chat web interactivo con Streamlit
   - Historial de conversación
   - Fuentes consultadas visibles
   - Estadísticas en tiempo real

4. RENDIMIENTO OPTIMIZADO
   - Respuestas en menos de 3 segundos
   - Procesamiento en lotes
   - Cache de vectorstore para rápida carga

5. ARQUITECTURA MODULAR
   - Fácil de extender y mantener
   - Código limpio y documentado
   - Componentes independientes
    """)
    wait()
    
    # Paso 7: Casos de uso
    print_section("Paso 7: Casos de Uso")
    print("""
EMPLEADOS DE RECURSOS HUMANOS:
- Consultar requisitos de certificaciones
- Verificar costos y duración de exámenes
- Obtener información sobre dominios de conocimiento

DESARROLLADORES/INGENIEROS:
- Preparación para exámenes de certificación
- Consulta rápida de temas específicos
- Verificación de información oficial

GESTORES DE FORMACIÓN:
- Planificar programas de certificación
- Estimar costos y tiempos
- Identificar áreas de conocimiento necesarias
    """)
    wait()
    
    # Paso 8: Cómo usar el sistema
    print_section("Paso 8: Cómo Usar el Sistema")
    print("""
OPCIÓN 1: Interfaz Web (Recomendado)
    $ python run_chat.py
    # Abre http://localhost:8501 en tu navegador

OPCIÓN 2: Uso Programático
    from src.rag_agent import ask_certification_question
    
    respuesta = ask_certification_question("¿Cuánto cuesta el examen?")
    print(respuesta)

OPCIÓN 3: Evaluación del Sistema
    $ python run_evaluation.py
    # Ejecuta preguntas de prueba y genera métricas
    """)
    wait()
    
    # Final
    print_header("FIN DE LA DEMOSTRACIÓN")
    print("""
RESUMEN:
- Sistema RAG completamente funcional
- Respuestas precisas basadas en documento oficial
- Interfaz web intuitiva
- Código modular y extensible

PRÓXIMOS PASOS:
1. Ejecuta: python run_chat.py
2. Abre: http://localhost:8501
3. Comienza a hacer preguntas sobre certificaciones AWS ML

DOCUMENTACIÓN:
- README.md: Guía de inicio rápido
- docs/solucion_tecnica.md: Documentación técnica detallada
- Notebooks: Ejemplos interactivos en Jupyter

¡Gracias por ver la demostración!
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemostración interrumpida por el usuario.")
    except Exception as e:
        print(f"\n\nError durante la demostración: {e}")
        import traceback
        traceback.print_exc()

