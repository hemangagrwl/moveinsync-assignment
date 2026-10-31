from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import jwt

from backfill.errors import Unauthorized

PBKDF2_ITERATIONS = 210_000


class Role(StrEnum):
    ADMIN = "ADMIN"  # everything in their tenant, including closing invoices
    OPERATOR = "OPERATOR"  # ingest, compute, inspect
    EMPLOYEE = "EMPLOYEE"  # read the distance of trips they rode in


@dataclass(frozen=True)
class Principal:
    username: str
    tenant_id: str
    role: Role
    employee_ref: str | None = None


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    salt_b64 = base64.b64encode(salt).decode()
    digest_b64 = base64.b64encode(digest).decode()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt_b64}${digest_b64}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iterations, salt_b64, digest_b64 = encoded.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt_b64), int(iterations))
    return hmac.compare_digest(digest, base64.b64decode(digest_b64))


# Used to spend the same time on unknown usernames as on wrong passwords.
_DUMMY_HASH = hash_password("timing-equaliser")


class UserStore:
    def __init__(self, users: dict[str, dict[str, Any]]) -> None:
        self._users = users

    @classmethod
    def from_file(cls, path: str | Path) -> UserStore:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({u["username"]: u for u in data["users"]})

    def authenticate(self, username: str, password: str) -> Principal:
        user = self._users.get(username)
        ok = verify_password(password, user["password_hash"] if user else _DUMMY_HASH)
        if user is None or not ok:
            raise Unauthorized("Invalid username or password.", code="INVALID_CREDENTIALS")
        return Principal(username, user["tenant_id"], Role(user["role"]), user.get("employee_ref"))


class TokenService:
    def __init__(self, secret: str, ttl_s: int = 3600) -> None:
        self._secret = secret
        self._ttl = ttl_s

    def issue(self, p: Principal) -> str:
        now = int(time.time())
        claims = {
            "sub": p.username,
            "tid": p.tenant_id,
            "role": p.role.value,
            "emp": p.employee_ref,
            "iat": now,
            "exp": now + self._ttl,
        }
        return jwt.encode(claims, self._secret, algorithm="HS256")

    def verify(self, token: str) -> Principal:
        try:
            c = jwt.decode(
                token, self._secret, algorithms=["HS256"], options={"require": ["exp", "sub", "tid"]}
            )
            return Principal(c["sub"], c["tid"], Role(c["role"]), c.get("emp"))
        except (jwt.PyJWTError, KeyError, ValueError) as exc:
            raise Unauthorized("Token is invalid or expired.", code="INVALID_TOKEN") from exc

    @property
    def ttl_s(self) -> int:
        return self._ttl
