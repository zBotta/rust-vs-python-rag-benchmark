"""Lightweight HTTP embedding server for the Rust pipeline.

Wraps sentence-transformers/all-MiniLM-L6-v2 and exposes a single endpoint:

    POST /embed
    Body:  {"texts": ["text1", "text2", ...]}
    Reply: {"embeddings": [[f32 x 384], ...]}

Start:  uv run python embedding_server.py [--port 8765]
Stop:   Ctrl-C or kill the process.

Uses waitress as the WSGI server for reliable Windows socket handling.
"""
from __future__ import annotations

import argparse
import json
import os

# ---------------------------------------------------------------------------
# SSL bypass — must happen before any network library is imported.
# ---------------------------------------------------------------------------
if os.environ.get("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ.update({
        "CURL_CA_BUNDLE": "",
        "REQUESTS_CA_BUNDLE": "",
        "SSL_CERT_FILE": "",
        "HF_HUB_DISABLE_SSL_VERIFICATION": "1",
        "PYTHONHTTPSVERIFY": "0",
    })
    try:
        import httpx
        _orig_client_init = httpx.Client.__init__
        _orig_async_init = httpx.AsyncClient.__init__

        def _patched_client_init(self, *args, **kwargs):
            if "verify" not in kwargs:
                kwargs["verify"] = False
            _orig_client_init(self, *args, **kwargs)

        def _patched_async_init(self, *args, **kwargs):
            if "verify" not in kwargs:
                kwargs["verify"] = False
            _orig_async_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_client_init  # type: ignore
        httpx.AsyncClient.__init__ = _patched_async_init  # type: ignore
    except ImportError:
        pass

from sentence_transformers import SentenceTransformer  # type: ignore
from waitress import serve  # type: ignore

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading model {_MODEL_NAME} ...", flush=True)
        _model = SentenceTransformer(_MODEL_NAME)
        print("Model loaded.", flush=True)
    return _model


def app(environ, start_response):
    """WSGI application."""
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "")

    if method == "GET" and path == "/health":
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"ok"]

    if method == "POST" and path == "/embed":
        try:
            length = int(environ.get("CONTENT_LENGTH", 0))
            body = environ["wsgi.input"].read(length)
            payload = json.loads(body)
            texts: list[str] = payload["texts"]
        except Exception as exc:
            start_response("400 Bad Request", [("Content-Type", "text/plain")])
            return [str(exc).encode()]

        model = get_model()
        vecs = model.encode(texts, normalize_embeddings=True).tolist()
        response_body = json.dumps({"embeddings": vecs}).encode()

        start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body))),
        ])
        return [response_body]

    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not found"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    # Eagerly load model so first request isn't slow
    get_model()

    print(f"Embedding server listening on http://127.0.0.1:{args.port}", flush=True)
    serve(app, host="127.0.0.1", port=args.port, threads=1)


if __name__ == "__main__":
    main()
