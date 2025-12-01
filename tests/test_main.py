import fastapi
import fastapi.testclient
import main


client = fastapi.testclient.TestClient(main.app)

def test_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "welcome" in data["message"].lower()
    assert isinstance(data["example_payload"], dict)


def test_inference_valid():
    response = client.post("/infer", json=main.example_payload)
    assert response.status_code == 200
    data = response.json()
    assert "salary" in data
    assert data["salary"] in (">50K", "<=50K")


# not sure how useful this is long-term, but sanity check wants me to have distinct test cases for each possible model output
def test_inference_valid_above_50k():
    payload = main.example_payload.copy()
    payload["sex"] = "Male"
    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "salary" in data
    assert data["salary"] == ">50K"


# not sure how useful this is long-term, but sanity check wants me to have distinct test cases for each possible model output
def test_inference_valid_below_50k():
    payload = main.example_payload.copy()
    payload["sex"] = "Female"
    response = client.post("/infer", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "salary" in data
    assert data["salary"] == "<=50K"


def test_inference_invalid_payload():
    invalid_payload = main.example_payload.copy()
    invalid_payload["sex"] = 55  # Invalid type
    response = client.post("/infer", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity due to validation error


if __name__ == "__main__":
    test_inference_invalid_payload()
