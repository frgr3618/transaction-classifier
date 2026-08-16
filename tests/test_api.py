from conftest import CATEGORIES

SAMPLE = "POS PURCH INR 1418 TXN9c72946c"


def test_root_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_predict_returns_200(client):
    assert client.post("/predict", json={"text": SAMPLE}).status_code == 200


def test_predict_returns_a_known_category(client):
    category = client.post("/predict", json={"text": SAMPLE}).json()["category"]
    assert category in CATEGORIES


def test_predict_is_deterministic(client):
    first = client.post("/predict", json={"text": SAMPLE}).json()
    second = client.post("/predict", json={"text": SAMPLE}).json()
    assert first == second


def test_predict_handles_unseen_text(client):
    """Out-of-vocabulary input must still return a category, not a 500."""
    response = client.post("/predict", json={"text": "qwertyuiop zxcvbnm"})
    assert response.status_code == 200
    assert response.json()["category"] in CATEGORIES


def test_predict_handles_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200
    assert response.json()["category"] in CATEGORIES


def test_predict_rejects_missing_field(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_wrong_type(client):
    assert client.post("/predict", json={"text": 123}).status_code == 422


def test_predict_response_shape(client):
    assert list(client.post("/predict", json={"text": SAMPLE}).json()) == ["category"]
