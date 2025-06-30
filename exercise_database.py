import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# modelo de embeddings
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# base de datos de ejercicios reales obtenidos de fuentes clinicas
EXERCISE_DB = [
    {
        "id": 1,
        "text": "Point to common objects in a picture (comprehension)",
        "type": "Broca",
        "severity": "Severe",
        "tags": ["comprehension", "objects"],
    },
    {
        "id": 2,
        "text": "Repeat simple words: 'water', 'bread', 'yes' (articulation)",
        "type": "Broca",
        "severity": "Severe",
        "tags": ["articulation", "repetition"],
    },
    {
        "id": 3,
        "text": "Name 5 body parts (naming)",
        "type": "Broca",
        "severity": "Moderate",
        "tags": ["naming", "categorization"],
    },
    {
        "id": 4,
        "text": "Complete simple sentences: 'I eat ______' (expression)",
        "type": "Broca",
        "severity": "Moderate",
        "tags": ["expression", "sentences"],
    },
    {
        "id": 5,
        "text": "Describe a sequence of daily actions (narration)",
        "type": "Broca",
        "severity": "Mild",
        "tags": ["narration", "sequences"],
    },
    {
        "id": 6,
        "text": "Answer yes/no questions about preferences (interaction)",
        "type": "Global",
        "severity": "Severe",
        "tags": ["comprehension", "interaction"],
    },
    {
        "id": 7,
        "text": "Match pictures with written words (reading)",
        "type": "Wernicke",
        "severity": "Moderate",
        "tags": ["reading", "association"],
    },
    {
        "id": 8,
        "text": "Explain the use of common objects (semantics)",
        "type": "Wernicke",
        "severity": "Mild",
        "tags": ["semantics", "explanation"],
    },
]

# genera embeddings para todos los ejercicios
exercise_embeddings = model.encode(
    [ex["text"] + " " + " ".join(ex["tags"]) for ex in EXERCISE_DB]
)

def retrieve_exercises(
    aphasia_type: str, severity: str, topic: str = "", failed_tags: list = [], n: int = 5
):
    """Recupera ejercicios relevantes usando RAG"""
    # Construir query enriquecida
    query = f"{aphasia_type} {severity} {topic}"
    if failed_tags:
        query += " con refuerzo en " + ", ".join(failed_tags)

    # Embedding de la query
    query_embedding = model.encode([query])[0]

    # Calcular similitud
    similarities = cosine_similarity([query_embedding], exercise_embeddings)[0]

    # Obtener índices ordenados por similitud
    sorted_indices = np.argsort(similarities)[::-1]

    # Filtrar por tipo, severidad y topic (si está presente en tags)
    filtered_exercises = []
    for idx in sorted_indices:
        ex = EXERCISE_DB[idx]
        if ex["type"] == aphasia_type and ex["severity"] == severity:
            if topic:
                if any(topic in tag.lower() for tag in ex["tags"]):
                    filtered_exercises.append(ex["text"])
            else:
                filtered_exercises.append(ex["text"])
        if len(filtered_exercises) >= n:
            break

    # Completar si no hay suficientes
    while len(filtered_exercises) < n:
        filtered_exercises.append("Ejercicio personalizado: " + query)

    return filtered_exercises[:n]
