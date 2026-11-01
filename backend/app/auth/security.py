"""Password hashing and token primitives (Phase 20). [OURS]

Standard library only - no new dependency is introduced for auth.

Passwords
---------
PBKDF2-HMAC-SHA256 with a per-user 16-byte salt and 390 000 iterations
(OWASP's 2023 recommendation for PBKDF2-HMAC-SHA256). Stored format::

    pbkdf2_sha256$<iterations>$<salt hex>$<digest hex>

Verification is constant-time (``hmac.compare_digest``). Plaintext
passwords are never stored, logged or echoed back.

Tokens
------
Opaque bearer tokens (``secrets.token_urlsafe(32)`` - 256 random bits).
The database stores only the SHA-256 digest of the token, so a stolen
database row cannot be replayed against the API. Comparison of digests
is constant-time as well.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

SCHEME = "pbkdf2_sha256"
DEFAULT_ITERATIONS = 390_000
_SALT_BYTES = 16


def hash_password(password: str, *, iterations: int = DEFAULT_ITERATIONS) -> str:
    """Return the stored digest for ``password`` (salted, iterated)."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return f"{SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time check of ``password`` against a stored digest."""
    if not isinstance(password, str) or not isinstance(stored, str):
        return False
    try:
        scheme, raw_iterations, salt_hex, digest_hex = stored.split("$")
        iterations = int(raw_iterations)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except (ValueError, AttributeError):
        return False
    if scheme != SCHEME or iterations < 1 or not salt or not expected:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(candidate, expected)


def new_token() -> str:
    """A fresh opaque bearer token (only its digest is ever persisted)."""
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    """SHA-256 digest of a bearer token, as hex."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
