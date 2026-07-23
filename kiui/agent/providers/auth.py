"""Durable provider credential storage under the user's personal kia directory."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from filelock import FileLock


@dataclass(frozen=True)
class OAuthCredential:
    access: str
    refresh: str
    expires: float
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: object) -> "OAuthCredential":
        if not isinstance(value, dict) or value.get("type") != "oauth":
            raise ValueError("Credential is not an OAuth credential")
        access = value.get("access")
        refresh = value.get("refresh")
        expires = value.get("expires")
        metadata = value.get("metadata", {})
        if not isinstance(access, str) or not access:
            raise ValueError("OAuth credential has no access token")
        if not isinstance(refresh, str) or not refresh:
            raise ValueError("OAuth credential has no refresh token")
        if not isinstance(expires, (int, float)):
            raise ValueError("OAuth credential has an invalid expiry")
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in metadata.items()
        ):
            raise ValueError("OAuth credential has invalid metadata")
        return cls(access, refresh, float(expires), dict(metadata))

    def to_json(self) -> dict:
        return {
            "type": "oauth",
            "access": self.access,
            "refresh": self.refresh,
            "expires": self.expires,
            "metadata": self.metadata,
        }


class CredentialStore:
    """Locked, atomic JSON credential store keyed by provider ID."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else Path.home() / ".kia" / "auth.json"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = FileLock(str(self.path) + ".lock")

    def _read_unlocked(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to read credential store {self.path}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"Credential store must contain a JSON object: {self.path}")
        return data

    def _write_unlocked(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        fd, staging_name = tempfile.mkstemp(prefix=".auth.", dir=self.path.parent)
        staging = Path(staging_name)
        try:
            os.chmod(staging, 0o600)
            with os.fdopen(fd, "wb") as file:
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(staging, self.path)
            os.chmod(self.path, 0o600)
            if os.name == "posix":
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            staging.unlink(missing_ok=True)

    def read_oauth(self, provider: str) -> OAuthCredential | None:
        with self.lock:
            value = self._read_unlocked().get(provider)
        return None if value is None else OAuthCredential.from_json(value)

    def write_oauth(self, provider: str, credential: OAuthCredential) -> None:
        with self.lock:
            data = self._read_unlocked()
            data[provider] = credential.to_json()
            self._write_unlocked(data)

    def modify_oauth(
        self,
        provider: str,
        update: Callable[[OAuthCredential | None], OAuthCredential],
    ) -> OAuthCredential:
        """Run a serialized read-modify-write, used for token refresh."""
        with self.lock:
            data = self._read_unlocked()
            value = data.get(provider)
            current = None if value is None else OAuthCredential.from_json(value)
            credential = update(current)
            data[provider] = credential.to_json()
            self._write_unlocked(data)
            return credential

    def delete(self, provider: str) -> bool:
        with self.lock:
            data = self._read_unlocked()
            if provider not in data:
                return False
            del data[provider]
            self._write_unlocked(data)
            return True
