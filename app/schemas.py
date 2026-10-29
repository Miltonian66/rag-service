from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DocumentIn(BaseModel):
    source: str = Field(min_length=1, description="Logical name/URI of the document.")
    text: str = Field(min_length=1, description="Raw document text to ingest.")


class DocumentOut(BaseModel):
    source: str
    chunks_ingested: int


class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    stream: bool = False
    mode: Literal["simple", "agentic"] = Field(
        default="simple",
        description="simple: retrieve → generate. agentic: LangGraph with grading, "
        "query rewriting and citation verification (no streaming).",
    )

    @model_validator(mode="after")
    def _agentic_is_not_streamed(self) -> "QueryIn":
        if self.mode == "agentic" and self.stream:
            raise ValueError("stream=true is only supported with mode=simple")
        return self


class Citation(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class QueryOut(BaseModel):
    answer: str
    citations: list[Citation]


class AgenticQueryOut(QueryOut):
    rewritten_query: str | None = None
    unsupported_citations: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)


class HealthOut(BaseModel):
    status: str
    embedder: str
    generator: str
