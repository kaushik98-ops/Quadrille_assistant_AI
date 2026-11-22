import pytest

from assistant.core.auth import Authenticator, credential_from_headers, key_id
from assistant.core.errors import ConfigError
from assistant.core.ratelimit import RateLimiter

K1, K2 = "a" * 20, "b" * 20


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_authenticator_rejects_missing_or_short_keys():
    with pytest.raises(ConfigError):
        Authenticator([])
    with pytest.raises(ConfigError):
        Authenticator(["short"])


def test_authenticator_identifies_keys():
    auth = Authenticator([K1, K2])
    assert auth.identify(K1) == key_id(K1)
    assert auth.identify(K2) == key_id(K2) != key_id(K1)
    assert auth.identify("wrong-key-0000000000") is None
    assert auth.identify("") is None and auth.identify(None) is None


def test_credential_from_headers():
    assert credential_from_headers("Bearer abc", None) == "abc"
    assert credential_from_headers("bearer   abc ", None) == "abc"
    assert credential_from_headers("Basic abc", None) is None
    assert credential_from_headers("Basic abc", "k") == "k"
    assert credential_from_headers(None, " k ") == "k"
    assert credential_from_headers("Bearer ", None) is None
    assert credential_from_headers(None, None) is None


def test_rate_limiter_window():
    clock = Clock()
    limiter = RateLimiter(2, window=60, clock=clock)
    assert limiter.check("k") == 0.0              # t=0
    clock.t = 10
    assert limiter.check("k") == 0.0
    clock.t = 20
    assert limiter.check("k") == pytest.approx(40.0)  # the t=0 hit leaves the window at t=60
    clock.t = 60
    assert limiter.check("k") == 0.0              # the first hit has expired
    clock.t = 61
    assert limiter.check("k") > 0                 # hits at 10 and 60 are still inside


def test_rate_limiter_is_per_key_and_can_be_disabled():
    limiter = RateLimiter(1, clock=Clock())
    assert limiter.check("a") == 0.0 and limiter.check("b") == 0.0
    assert limiter.check("a") > 0
    off = RateLimiter(0)
    assert all(off.check("a") == 0.0 for _ in range(100))
