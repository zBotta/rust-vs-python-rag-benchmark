"""Download the Wikipedia parquet file for the Rust pipeline's local fallback."""
import os, pathlib, shutil

# SSL bypass
if os.environ.get("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ.update({"CURL_CA_BUNDLE": "", "REQUESTS_CA_BUNDLE": "", "SSL_CERT_FILE": "", "HF_HUB_DISABLE_SSL_VERIFICATION": "1"})
    import httpx
    _orig = httpx.Client.__init__
    def _p(self, *a, **kw): kw.setdefault("verify", False); _orig(self, *a, **kw)
    httpx.Client.__init__ = _p  # type: ignore

from huggingface_hub import hf_hub_download

subset = "20231101.simple"
dest_dir = pathlib.Path("data") / subset
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading parquet to {dest_dir} ...")
cached = hf_hub_download(
    repo_id="wikimedia/wikipedia",
    repo_type="dataset",
    filename=f"{subset}/train-00000-of-00001.parquet",
)
dest = dest_dir / "train-00000-of-00001.parquet"
shutil.copy2(cached, dest)
print(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
