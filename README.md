# Afasia LLM: Generador de ejercicios terapéuticos personalizados

Este proyecto implementa un sistema de generación automática de ejercicios de lenguaje adaptados a pacientes con afasia. Combina un enfoque híbrido que integra un banco estructurado de plantillas con generación mediante modelos de lenguaje (LLM), ajustado por contexto clínico (tipo y severidad de afasia) y tópico temático.

## Estructura del repositorio

```
afasia_llm/
├── exercise_bank.py              # Banco de ejercicios estructurado por severidad y tópico
├── exercise_database.py         # Recuperación de ejercicios reales con embeddings (RAG)
├── generate.py                  # Función principal de generación (LLM + fallback)
├── terapeuta_virtual.py         # Interfaz Gradio
│
├── notebook/
    └── quality_metrics.ipynb    # Evaluación de calidad de los ejercicios generados
```

## Componentes

### generate.py

- Función principal para generar ejercicios (`generate_exercises_llm`) con:
  - RAG (ejemplos reales recuperados por embeddings)
  - Prompt instructivo a un modelo LLM (`Flan-T5`)
  - Fallback a plantillas locales si la generación falla
- Controla temperatura, do_sample, top_p, etc.
- Incluye validación heurística (`is_valid_exercise`)

### exercise_bank.py

- Diccionario estructurado (`EXERCISE_BANK`) con ejercicios divididos por:
  - Severidad (`Severe`, `Moderate`, `Mild`)
  - Tópico (`daily activities`, `shopping`, etc.)
- Se usa como fallback cuando no hay suficientes respuestas válidas del modelo

### exercise_database.py

- Recuperación semántica de ejercicios similares mediante `sentence-transformers`
- Permite la lógica RAG para alimentar el prompt con ejemplos contextualizados

### terapeuta_virtual.py

- Interfaz Gradio funcional para probar el sistema
- Parámetros configurables: severidad, tipo de afasia, tópico
- Permite ajustar el número de ejercicios y ver fallback

### notebook/quality_metrics.ipynb

- Evaluación de calidad con métricas heurísticas y análisis semántico
- Agrupación de outputs válidos vs. no válidos
- Comparación entre generación pura y fallback

## Requisitos

Modelos necesarios disponibles en HuggingFace:

- google/flan-t5-base
- sentence-transformers/all-MiniLM-L6-v2

## Ejemplo de uso por terminal

```
python generate.py --llm
--severity "Moderate"
--type "Broca"
--topic "shopping"
-n 5
```
