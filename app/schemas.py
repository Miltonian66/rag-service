from pydantic import BaseModel, Field


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


class Citation(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class QueryOut(BaseModel):
    answer: str
    citations: list[Citation]


class HealthOut(BaseModel):
    status: str
    embedder: str
    generator: str
