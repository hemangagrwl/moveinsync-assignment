"""Writes deploy/users.json with demo accounts for two tenants.

These are local demo credentials for the take-home only. In production users come from the
identity provider and this file does not exist.
"""

from __future__ import annotations

import json
from pathlib import Path

from backfill.api.security import hash_password

DEMO_USERS = [
    ("acme-admin", "acme", "ADMIN", None, "acme-admin-pw"),
    ("acme-ops", "acme", "OPERATOR", None, "acme-ops-pw"),
    ("acme-rider", "acme", "EMPLOYEE", "E00042", "acme-rider-pw"),
    ("globex-admin", "globex", "ADMIN", None, "globex-admin-pw"),
    ("globex-ops", "globex", "OPERATOR", None, "globex-ops-pw"),
]


def main() -> None:
    users = [
        {"username": u, "tenant_id": t, "role": r, "employee_ref": e, "password_hash": hash_password(pw)}
        for u, t, r, e, pw in DEMO_USERS
    ]
    out = Path(__file__).resolve().parents[1] / "deploy" / "users.json"
    out.write_text(json.dumps({"users": users}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(users)} users to {out}")


if __name__ == "__main__":
    main()
