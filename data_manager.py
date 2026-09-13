"""Small JSON persistence layer for the local prototype."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / "runtime"
SESSIONS_FILE = RUNTIME_DIR / "sessions.json"
_lock = threading.RLock()


class DataStore:
    """Persist completed sessions with atomic writes in one application process."""

    _DEFAULT = {"sessions": []}

    def initialize(self) -> None:
        with _lock:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            if not SESSIONS_FILE.exists():
                self._atomic_write(self._DEFAULT)
            else:
                self._read()

    def _read(self) -> dict[str, Any]:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        if not SESSIONS_FILE.exists():
            self._atomic_write(self._DEFAULT)
            return {"sessions": []}
        try:
            with SESSIONS_FILE.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
                raise ValueError("Invalid session store shape")
            return data
        except (json.JSONDecodeError, OSError, ValueError):
            backup = SESSIONS_FILE.with_suffix(".corrupt.json")
            if SESSIONS_FILE.exists():
                os.replace(SESSIONS_FILE, backup)
            self._atomic_write(self._DEFAULT)
            return {"sessions": []}

    def _atomic_write(self, data: dict[str, Any]) -> None:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        temporary = SESSIONS_FILE.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, SESSIONS_FILE)

    def append_session(self, session: dict[str, Any]) -> None:
        with _lock:
            data = self._read()
            data["sessions"].append(session)
            self._atomic_write(data)

    def get_all_sessions(self) -> list[dict[str, Any]]:
        with _lock:
            return list(self._read()["sessions"])

    def get_latest(self) -> dict[str, Any] | None:
        sessions = self.get_all_sessions()
        return sessions[-1] if sessions else None

    def count(self) -> int:
        return len(self.get_all_sessions())
