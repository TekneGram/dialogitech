from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .models import ResearchSessionState


class ResearchSessionInitializer:
    def initialize(self, *, session_path: str | Path, root_query: str) -> ResearchSessionState:
        timestamp = self._utc_now()
        return ResearchSessionState(
            session_id=self._session_id_from_path(session_path),
            created_at=timestamp,
            updated_at=timestamp,
            root_query=root_query,
        )

    def _session_id_from_path(self, session_path: str | Path) -> str:
        return Path(session_path).stem

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
