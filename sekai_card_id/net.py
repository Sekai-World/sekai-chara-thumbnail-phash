from __future__ import annotations

import time
import urllib.error
import urllib.request

USER_AGENT = "sekai-card-id/0.1 (+https://github.com/Sekai-World/sekai-chara-thumbnail-phash)"


class FetchError(RuntimeError):
    pass


def fetch_bytes(url: str, timeout: float = 30, retries: int = 3) -> bytes | None:
    """GET a URL. Returns None when the resource does not exist (404 / missing file).

    Transient failures are retried with exponential backoff and then raised
    as FetchError, so a flaky network is never mistaken for a missing asset.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                return None
            err: Exception = e
        except urllib.error.URLError as e:
            if isinstance(e.reason, FileNotFoundError):  # file:// URLs
                return None
            err = e
        except (TimeoutError, ConnectionError) as e:
            err = e
        if attempt < retries:
            time.sleep(2**attempt)
    raise FetchError(f"{url}: {err}")
