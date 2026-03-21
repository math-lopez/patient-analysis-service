import json
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama

load_dotenv()

app = FastAPI(title="CliniNotes Patient Analysis Service")

INTERNAL_TOKEN = os.getenv("PROCESSING_SERVICE_INTERNAL_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.gguf")
N_CTX = int(os.getenv("MODEL_CONTEXT", "8192"))
N_THREADS = int(os.getenv("MODEL_THREADS", "4"))

if not INTERNAL_TOKEN:
    raise RuntimeError("PROCESSING_SERVICE_INTERNAL_TOKEN não configurado")

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_threads=4,
            verbose=False,
        )
    return _llm


class SessionInput(BaseModel):
    id: str
    sessionDate: str | None = None
    manualNotes: str | None = None
    transcript: str | None = None
    highlights: list[str] = Field(default_factory=list)
    nextSteps: str | None = None


class PatientInput(BaseModel):
    id: str
    name: str


class AnalysisRequest(BaseModel):
    patientId: str
    psychologistId: str
    patient: PatientInput
    sessions: list[SessionInput]


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _llm is not None}


def build_prompt(payload: AnalysisRequest) -> str:
    sessions_text = []

    for idx, s in enumerate(payload.sessions, start=1):
        sessions_text.append(
            f"""
Sessão {idx}
Data: {s.sessionDate or ""}
Notas manuais: {s.manualNotes or ""}
Transcrição: {s.transcript or ""}
Highlights: {", ".join(s.highlights or [])}
Próximos passos: {s.nextSteps or ""}
""".strip()
        )

    joined_sessions = "\n\n".join(sessions_text)

    return f"""
Você é um assistente de apoio clínico administrativo.
Sua tarefa é gerar uma sugestão de síntese longitudinal com base apenas nos dados fornecidos.
Não invente fatos.
Não dê diagnóstico fechado.
Não trate a resposta como decisão clínica final.
Se faltar informação, diga isso explicitamente.

Responda SOMENTE em JSON válido com esta estrutura:
{{
  "summary": "texto",
  "recommendations": ["item 1", "item 2"],
  "risk_flags": ["item 1"],
  "data_gaps": ["item 1"]
}}

Paciente: {payload.patient.name}
ID do paciente: {payload.patient.id}

Histórico de sessões:
{joined_sessions}
""".strip()


@app.post("/process-patient-analysis")
async def process_patient_analysis(
    payload: AnalysisRequest,
    authorization: str | None = Header(default=None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization inválido")

    token = authorization.replace("Bearer ", "", 1)
    if token != INTERNAL_TOKEN:
        raise HTTPException(status_code=403, detail="Token interno inválido")

    if not payload.sessions:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Nenhuma sessão disponível para análise"
            },
        )

    try:
        llm = get_llm()
        prompt = build_prompt(payload)

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Responda apenas com JSON válido."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        return {
            "success": True,
            "summary": parsed.get("summary", ""),
            "recommendations": parsed.get("recommendations", []),
            "risk_flags": parsed.get("risk_flags", []),
            "raw_response": parsed,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )