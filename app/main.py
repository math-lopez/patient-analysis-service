import json
import os
import traceback
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
N_CTX = int(os.getenv("MODEL_CONTEXT", "1024"))
N_THREADS = int(os.getenv("MODEL_THREADS", "2"))
MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "120"))
TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.1"))

if not INTERNAL_TOKEN:
    raise RuntimeError("PROCESSING_SERVICE_INTERNAL_TOKEN não configurado")

_llm = None


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        print(f"[analysis] carregando modelo: {MODEL_PATH}", flush=True)
        print(
            f"[analysis] n_ctx={N_CTX} n_threads={N_THREADS} max_tokens={MAX_TOKENS}",
            flush=True,
        )
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            verbose=False,
        )
        print("[analysis] modelo carregado com sucesso", flush=True)
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
    return {
        "status": "ok",
        "model_loaded": _llm is not None,
        "model_path": MODEL_PATH,
        "n_ctx": N_CTX,
        "max_tokens": MAX_TOKENS,
    }


def build_prompt(payload: AnalysisRequest) -> str:
    sessions_text = []

    for idx, s in enumerate(payload.sessions, start=1):
        sessions_text.append(
            f"""
Sessão {idx}
ID: {s.id}
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
Sua tarefa é gerar uma síntese longitudinal objetiva com base SOMENTE nas informações fornecidas.

Regras obrigatórias:
- Não invente fatos
- Não dê diagnóstico fechado
- Não trate a resposta como decisão clínica final
- Se faltar informação, diga isso explicitamente
- Responda SOMENTE com JSON válido
- Não use markdown
- Não use crases
- Não escreva explicações fora do JSON
- Seja direto e objetivo

Formato obrigatório de saída:
{{
  "summary": "resumo longitudinal objetivo",
  "recommendations": ["recomendação 1", "recomendação 2"],
  "risk_flags": ["alerta 1"],
  "data_gaps": ["lacuna 1"]
}}

Paciente: {payload.patient.name}
ID do paciente: {payload.patient.id}

Histórico de sessões:
{joined_sessions}

Responda agora SOMENTE com JSON válido:
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()

    if not text:
        raise ValueError("Modelo retornou conteúdo vazio.")

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    if start == -1:
        raise ValueError(f"Modelo não retornou JSON válido. Saída bruta: {repr(text[:500])}")

    brace_count = 0
    end = -1

    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i
                break

    if end == -1:
        raise ValueError(f"Não foi possível encontrar um JSON completo. Saída bruta: {repr(text[:500])}")

    json_text = text[start:end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Falha ao fazer parse do JSON. Erro: {e}. JSON bruto: {repr(json_text[:500])}"
        ) from e


def normalize_output(parsed: dict[str, Any]) -> dict[str, Any]:
    summary = parsed.get("summary", "")
    recommendations = parsed.get("recommendations", [])
    risk_flags = parsed.get("risk_flags", [])
    data_gaps = parsed.get("data_gaps", [])

    if not isinstance(summary, str):
        summary = str(summary)

    if not isinstance(recommendations, list):
        recommendations = [str(recommendations)] if recommendations else []

    if not isinstance(risk_flags, list):
        risk_flags = [str(risk_flags)] if risk_flags else []

    if not isinstance(data_gaps, list):
        data_gaps = [str(data_gaps)] if data_gaps else []

    recommendations = [str(item) for item in recommendations]
    risk_flags = [str(item) for item in risk_flags]
    data_gaps = [str(item) for item in data_gaps]

    return {
        "summary": summary,
        "recommendations": recommendations,
        "risk_flags": risk_flags,
        "data_gaps": data_gaps,
    }


def call_model_completion(llm: Llama, prompt: str) -> str:
    print("[analysis] tentando create_completion...", flush=True)

    response = llm.create_completion(
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=["```", "\n\nPaciente:", "\n\nHistórico de sessões:"],
    )

    print("[analysis] create_completion retornou", flush=True)

    choices = response.get("choices", [])
    if not choices:
        raise ValueError("Modelo não retornou choices no create_completion.")

    content = (choices[0].get("text") or "").strip()

    print("[analysis] conteúdo bruto do completion:", flush=True)
    print(repr(content[:1000]), flush=True)

    if not content:
        raise ValueError("Modelo retornou resposta vazia no create_completion.")

    return content


def generate_analysis(llm: Llama, prompt: str) -> dict[str, Any]:
    raw_text = call_model_completion(llm, prompt)
    parsed = extract_json(raw_text)
    return parsed


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
                "error": "Nenhuma sessão disponível para análise",
            },
        )

    try:
        print("[analysis] requisição recebida", flush=True)
        print(f"[analysis] patientId={payload.patientId}", flush=True)
        print(f"[analysis] psychologistId={payload.psychologistId}", flush=True)
        print(f"[analysis] sessions={len(payload.sessions)}", flush=True)

        llm = get_llm()
        print("[analysis] modelo pronto", flush=True)

        prompt = build_prompt(payload)
        print(f"[analysis] prompt montado, tamanho={len(prompt)} caracteres", flush=True)
        print("[analysis] chamando modelo...", flush=True)

        parsed = generate_analysis(llm, prompt)
        normalized = normalize_output(parsed)

        print("[analysis] JSON parseado com sucesso", flush=True)
        print("[analysis] retornando resposta final", flush=True)

        return {
            "success": True,
            "summary": normalized["summary"],
            "recommendations": normalized["recommendations"],
            "risk_flags": normalized["risk_flags"],
            "data_gaps": normalized["data_gaps"],
            "raw_response": parsed,
        }

    except Exception as e:
        print("[analysis] erro:", flush=True)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )