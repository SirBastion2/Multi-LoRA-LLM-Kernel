"""FastAPI demo stub — multi-pane chat API without loading Mistral on scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Persona-Serve Demo (stub)")


class ChatRequest(BaseModel):
    persona_id: str
    message: str


@dataclass
class EngineStats:
    tokens_per_sec: float = 0.0
    batch_size: int = 0
    distinct_adapters: int = 0
    resident_slots: int = 0


_stats = EngineStats(resident_slots=0)
_personas: Dict[str, str] = {
    "alpha": "Persona Alpha (stub)",
    "beta": "Persona Beta (stub)",
    "gamma": "Persona Gamma (stub)",
}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PATH.read_text()


@app.get("/personas")
def list_personas() -> List[str]:
    return list(_personas.keys())


@app.get("/stats")
def stats() -> EngineStats:
    return _stats


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    async def stream() -> AsyncIterator[bytes]:
        label = _personas.get(req.persona_id, req.persona_id)
        reply = (
            f"[stub] {label} received: {req.message!r}. "
            "Wire StaticBatchEngine + Mistral on owner GPU to stream real tokens."
        )
        for word in reply.split():
            yield (word + " ").encode("utf-8")

    return StreamingResponse(stream(), media_type="text/plain")


HTML_PATH = __import__("pathlib").Path(__file__).with_name("index.html")
