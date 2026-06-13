from fastapi import APIRouter

from app.dependencies import SettingsDep
from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health(settings: SettingsDep) -> HealthOut:
    return HealthOut(
        status="ok",
        embedder=settings.embedding_provider,
        generator=settings.llm_provider,
    )
