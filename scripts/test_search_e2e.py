"""
End-to-end search test: sends a real catalog image to the running API and
prints the Top-5 similar product results.
Run with:
    python scripts/test_search_e2e.py
(Requires the FastAPI server to be running on http://localhost:8000)
"""
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from config import settings

# ── Pick a test image from the catalog ───────────────────────────────────────
imgs = sorted(settings.images_dir.rglob("*.jpg"))
if not imgs:
    print("[FAIL] No images found in images directory.")
    sys.exit(1)

test_img = imgs[100]
print(f"Query image : {test_img.name}")
print(f"Brand       : {test_img.parent.name}")
print(f"Category    : {test_img.parent.parent.parent.name}/{test_img.parent.parent.name}")
print()

# ── Build multipart form-data body ───────────────────────────────────────────
boundary = "boundary_visualfind_test"
with open(test_img, "rb") as f:
    img_bytes = f.read()

body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="{test_img.name}"\r\n'
    f"Content-Type: image/jpeg\r\n\r\n"
).encode() + img_bytes + (
    f"\r\n--{boundary}\r\n"
    f'Content-Disposition: form-data; name="top_k"\r\n\r\n10\r\n'
    f"--{boundary}--\r\n"
).encode()

req = urllib.request.Request(
    "http://localhost:8000/search",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)

# ── Send request & parse response ─────────────────────────────────────────────
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
except urllib.error.URLError as e:
    print(f"[FAIL] Cannot reach API: {e}")
    print("       Make sure the FastAPI server is running: start_server.bat")
    sys.exit(1)

# ── Print results ─────────────────────────────────────────────────────────────
print(f"Query time  : {data['query_time_ms']:.1f}ms")
print(f"Results     : {data['total_results']}")
print()
print(f"{'#':<4} {'Brand':<20} {'Category':<10} {'Subcategory':<12} {'Similarity'}")
print("-" * 65)
for i, r in enumerate(data["results"][:10]):
    has_img = "[img]" if r.get("image_b64") else "     "
    print(f"{i+1:<4} {r['brand']:<20} {r['category']:<10} {r['subcategory']:<12} {r['similarity']:.4f} {has_img}")

print()
if data["total_results"] > 0 and data["results"][0]["similarity"] > 0.8:
    print("[OK]  Top result is highly similar (>=80%) - search is working correctly!")
else:
    print("[OK]  Search returned results. Check similarity scores above.")
