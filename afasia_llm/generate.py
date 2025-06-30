from __future__ import annotations
import random, re, warnings

warnings.filterwarnings("ignore")

from exercise_database import retrieve_exercises
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from exercise_bank import EXERCISE_BANK

def _sample_bank(sev: str, topic: str, k: int) -> list[str]:
    sev = sev.title()
    if sev not in EXERCISE_BANK:                       # fallback
        sev = "Moderate"
    topics = EXERCISE_BANK[sev]
    # elige el tópico con mayor solape de palabras
    best_topic = max(topics, key=lambda t: sum(w in topic.lower() for w in t.split()))
    pool = topics[best_topic]
    return random.sample(pool, min(k, len(pool)))

# validación heurística
THERA = "point name describe complete tell say answer explain touch show nod wave ask list compare give plan debate".split()
NON = "red apple screen black white level test side confused correct way".split()

def is_therapeutic(txt: str) -> bool:
    low = txt.lower()
    return (
        any(k in low for k in THERA)
        and not any(b in low for b in NON)
        and len(txt.split()) >= 4
    )

def is_valid_exercise(txt: str) -> bool:
    return is_therapeutic(txt)

# plantillas locales
def generate_exercises_local(severity, aphasia_type, topic, n=5) -> list[str]:
    cand = _sample_bank(severity, topic or aphasia_type, n * 2)
    valid = [c for c in cand if is_valid_exercise(c)]
    while len(valid) < n:
        e = random.choice(cand)
        if e not in valid:
            valid.append(e)
    random.shuffle(valid)
    return valid[:n]

# LLM + RAG
_LLM_CACHE = {"model": None, "tok": None}

def _load_llm():
    if _LLM_CACHE["model"] is not None:
        return _LLM_CACHE["model"], _LLM_CACHE["tok"]
    try:
        name = "google/flan-t5-base"
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(name)
        _LLM_CACHE["model"], _LLM_CACHE["tok"] = mdl, tok
        return mdl, tok
    except Exception as e:
        warnings.warn(f"[generate] LLM OFF ({e}) → uso plantillas.")
        return None, None

def _call_flan(prompt: str, max_new=128) -> str | None:
    mdl, tok = _load_llm()
    if mdl is None:
        return None
    outs = mdl.generate(
        **tok(prompt, return_tensors="pt"),
        max_new_tokens=max_new,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
    )
    return tok.decode(outs[0], skip_special_tokens=True)

def _clean_llm(txt: str) -> list[str]:
    if not txt or "cancel" in txt.lower():
        return []
    parts = re.split(r"\n|•|-|\d+\.", txt)
    return [p.strip(" -\t") for p in parts if is_valid_exercise(p)]

def generate_exercises_llm(severity, aphasia_type, topic, n=5, max_tries=4):
    rag_examples = retrieve_exercises(aphasia_type, severity, topic=topic, n=3)
    demo = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(rag_examples, 1))

    prompt = (
        "You are a certified speech-language therapist.\n"
        f"Aphasia type: {aphasia_type}\n"
        f"Patient severity: {severity} aphasia.\n"
        f"Topic: {topic}.\n\n"
        "Write exactly 5 short therapeutic language exercises (≤20 words). "
        "Use imperative verbs and number them 1-5.\n\n"
        "### Few-shot examples\n" + demo + "\n### Your turn\n"
    )

    collected, tries = [], 0
    while len(collected) < n and tries < max_tries:
        tries += 1
        gen = _clean_llm(_call_flan(prompt) or "")
        collected += [g for g in gen if g not in collected]

    if len(collected) < n:
        collected += generate_exercises_local(severity, aphasia_type, topic, n - len(collected))

    return collected[:n]

if __name__ == "__main__":
    import argparse, textwrap

    ap = argparse.ArgumentParser()
    ap.add_argument("--severity", default="Moderate")
    ap.add_argument("--type", dest="aphasia_type", default="Broca")
    ap.add_argument("--topic", default="daily activities")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    f = generate_exercises_llm if a.llm else generate_exercises_local
    ex = f(a.severity, a.aphasia_type, a.topic, n=a.n)
    print("\n".join(textwrap.wrap(" | ".join(ex), 100)))
