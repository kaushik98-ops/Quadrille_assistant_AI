"""API-key authentication. Keys come from config and are never logged."""
import hashlib
import hmac

from assistant.core.errors import ConfigError

MIN_KEY_LENGTH = 16


def key_id(key: str) -> str:
    """A short, non-reversible label for a key. Safe to log and to use as a namespace."""
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def credential_from_headers(authorization: str | None, x_api_key: str | None) -> str | None:
    """Accepts 'Authorization: Bearer <key>' or 'X-API-Key: <key>'."""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    return None


class Authenticator:
    def __init__(self, keys: list[str]):
        if not keys:
            raise ConfigError("No API keys configured. Set API_KEYS in your .env file (see .env.example).")
        if any(len(k) < MIN_KEY_LENGTH for k in keys):
            raise ConfigError(f"Every API key must be at least {MIN_KEY_LENGTH} characters long.")
        self._keys = [(k.encode(), key_id(k)) for k in keys]

    def identify(self, credential: str | None) -> str | None:
        """Return the key's id if the credential matches a configured key, else None."""
        if not credential:
            return None
        candidate = credential.encode()
        found = None
        for key, kid in self._keys:  # compare against all keys, in constant time per key
            if hmac.compare_digest(candidate, key):
                found = kid
        return found
