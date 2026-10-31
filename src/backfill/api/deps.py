from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backfill.api.security import Principal, Role
from backfill.container import Container
from backfill.errors import Forbidden, Unauthorized

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def current_principal(
    c: ContainerDep,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if creds is None or creds.scheme.lower() != "bearer":
        raise Unauthorized()
    return c.tokens.verify(creds.credentials)


def require(*roles: Role) -> Callable[[Principal], Principal]:
    allowed = frozenset(roles)

    def check(p: Annotated[Principal, Depends(current_principal)]) -> Principal:
        if p.role not in allowed:
            raise Forbidden()
        return p

    return check


AnyUser = Annotated[Principal, Depends(require(Role.ADMIN, Role.OPERATOR, Role.EMPLOYEE))]
Staff = Annotated[Principal, Depends(require(Role.ADMIN, Role.OPERATOR))]
Admin = Annotated[Principal, Depends(require(Role.ADMIN))]
