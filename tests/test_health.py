import requests, os

API_URL = os.getenv("API_URL", "http://localhost:8000")

def test_health():
    r = requests.get(f"{API_URL}/health", timeout=10)
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
