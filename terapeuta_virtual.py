import re, gradio as gr
from typing import Optional
from generate import _load_llm
from generate import generate_exercises_llm

SEVS = {"mild", "moderate", "severe"}
TOPICS = ["daily activities", "hygiene", "shopping", "transport", "family", "emotions"]

def _norm(s: str) -> Optional[str]:
    t = s.strip().lower()
    return t.title() if t in SEVS else None

def parse(raw: str):
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        return None, "Formato: tipo, severidad   ej:  Broca, Severe"
    typ, sev = parts
    sevOk = _norm(sev)
    if not sevOk:
        return None, "Severidad debe ser Mild / Moderate / Severe"
    return dict(aphasia_type=typ.title(), severity=sevOk), "Generando 5 ejercicios…"

def pretty(xs):
    return "\n".join(f"{i+1}. {x}" for i, x in enumerate(xs))

class Session:
    def __init__(self):
        self.info = None
        self.exercises = []
        self.topic = "daily activities"

    def _gen(self, n=5):
        s, t = self.info["severity"], self.info["aphasia_type"]
        return generate_exercises_llm(s, t, self.topic, n=n)

    def init(self):
        self.exercises = self._gen()

    def adapt(self, idx):
        for i in idx:
            self.exercises[i] = self._gen(n=1)[0]

def chat(msg, hist, ses: Optional[Session], topic):
    msg = msg.strip()
    if ses is None:
        ses, hist = Session(), []
    ses.topic = topic  # actualizar el tema desde el dropdown

    if ses.info is None:
        info, resp = parse(msg)
        if info is None:
            return resp, hist, ses
        ses.info = info
        ses.init()
        resp += "\n\n" + pretty(ses.exercises)
        return "", hist + [(msg, resp)], ses

    idx = [int(x) - 1 for x in re.findall(r"\d+", msg) if 1 <= int(x) <= 5]
    if not idx:
        return "Indica nº fallados (1-5)", hist, ses
    ses.adapt(idx)
    resp = "Ejercicios adaptados:\n\n" + pretty(ses.exercises)
    return "", hist + [(msg, resp)], ses

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Asistente de Terapia de Lenguaje para Afasia ")

    cb = gr.Chatbot(height=330)
    inp = gr.Textbox(placeholder="Broca, Severe")
    topic_dropdown = gr.Dropdown(
        choices=TOPICS, value="daily activities", label="Tema de los ejercicios"
    )
    state = gr.State()

    inp.submit(chat, [inp, cb, state, topic_dropdown], [inp, cb, state])
    gr.Examples(["Broca, Severe", "1 3"], inputs=inp)

if __name__ == "__main__":
    _load_llm()  # Flan-T5
    demo.launch(server_name="0.0.0.0", server_port=3000)
