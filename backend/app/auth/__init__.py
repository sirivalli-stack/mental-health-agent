"""Authentication, ownership and user data (Phase 20). [OURS]

This package adds the *minimum* authentication layer the product goals
require, without touching the research pipeline:

* :mod:`app.auth.security` - password hashing (PBKDF2-HMAC-SHA256, stdlib
  only) and opaque bearer tokens (only their SHA-256 digest is stored);
* :mod:`app.auth.store`   - users, tokens, conversation ownership, stored
  messages and per-user preferences in the same SQLite file the project
  already uses (``DATABASE_PATH``, default ``data/sessions.db``);
* :mod:`app.auth.deps`    - FastAPI dependencies resolving the caller.

Privacy posture:

* passwords are stored only as salted PBKDF2 digests - never plaintext,
  never reversible;
* bearer tokens are random 256-bit values; the database keeps only their
  SHA-256 digest, so a database leak cannot be replayed as a login;
* conversation *messages* are stored (the product requires reopening
  history, which Phase 7 snapshots cannot provide - they keep only the
  recent user turns) - they are scoped to their owner and never returned
  to another user;
* the Phase 15 turn audit stays metadata-only and unchanged.
"""

from app.auth.security import (  # noqa: F401
    hash_password,
    new_token,
    token_digest,
    verify_password,
)
