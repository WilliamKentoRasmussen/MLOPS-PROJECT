from fastapi.testclient import TestClient

from src.main_project.backend import app

client = TestClient(app)
def test_read_root():
    """Test root endpoint returns correct API information."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    # Check structure of response
    assert "message" in data
    assert "version" in data
    assert "endpoints" in data

    # Check endpoints list
    endpoints = data["endpoints"]
    expected_endpoints = ["docs", "health", "models", "predict", "predict_batch"]
    for endpoint in expected_endpoints:
        assert endpoint in endpoints

def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    # Basic health check
    assert data["status"] == "healthy"
    assert "device" in data
    assert "available_models" in data

    # Models directory info (exists might be True or False)
    assert "models_directory" in data
    assert "models_exist" in data  # This could be True or False!
    assert "cached_models" in data
    assert isinstance(data["cached_models"], list)

def test_list_models():
    """Test models listing endpoint."""
    response = client.get("/models")

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "available_models" in data
    assert isinstance(data["available_models"], list)
    assert "models" in data
    assert isinstance(data["models"], dict)
    assert "loaded_models" in data
    assert isinstance(data["loaded_models"], list)
    assert "directory_exists" in data  # This is the correct key!
    assert "models_directory" in data
    # Check that expected models are listed
    expected_models = ["baseline", "alexnet", "vgg16"]
    for model in expected_models:
        assert model in data["available_models"]
        assert model in data["models"]

        # Check model info structure
        model_info = data["models"][model]
        assert "path" in model_info
        assert "exists" in model_info
        assert "loaded" in model_info

def test_model_info_default():
    """Test model info endpoint with default model."""
    response = client.get("/info")

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "model_name" in data
    assert data["model_name"] == "baseline"  # Default
    assert "num_classes" in data
    assert data["num_classes"] == 2
    assert "class_names" in data
    assert data["class_names"] == ["NORMAL", "PNEUMONIA"]
    assert "device" in data
    assert "model_path" in data
    assert "file_exists" in data







