from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


def create_engine_and_sessionmaker(
    settings: Settings,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(settings.database_url, future=True, pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    return engine, factory


# Lazily-initialised module singletons. The real app uses these; tests never touch
# the DB because get_session is overridden via dependency_overrides.
_settings = get_settings()
engine, AsyncSessionLocal = create_engine_and_sessionmaker(_settings)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session
