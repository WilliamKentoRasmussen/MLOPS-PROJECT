import random
from pathlib import Path

from locust import HttpUser, between, task

# Directory containing sample images for testing
SAMPLE_IMAGES_DIR = Path("tests/sample_images")  # Put a few JPG/PNG images here
sample_files = list(SAMPLE_IMAGES_DIR.glob("*.jpg")) + list(SAMPLE_IMAGES_DIR.glob("*.png"))

class XRayUser(HttpUser):
    """
    Locust user class for Chest X-Ray FastAPI API.
    Simulates realistic API usage:
    - Health checks
    - Listing models
    - Single predictions
    - Batch predictions
    """
    wait_time = between(1, 3)  # random wait between tasks

    @task(2)
    def health_check(self):
        """Hit the /health endpoint"""
        self.client.get("/health", name="/health")

    @task(2)
    def list_models(self):
        """Hit the /models endpoint"""
        self.client.get("/models", name="/models")

    @task(5)
    def predict_single(self):
        """Send a single X-Ray image to /predict"""
        if not sample_files:
            return
        file_path = random.choice(sample_files)
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "image/jpeg")}
            with self.client.post(
                "/predict?model=baseline",
                files=files,
                name="/predict",
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Failed: {response.status_code}")

    @task(3)
    def predict_batch(self):
        """Send multiple X-Ray images to /predict-batch"""
        if len(sample_files) < 2:
            return
        selected_files = random.sample(sample_files, min(3, len(sample_files)))
        files_payload = [
            ("files", (file_path.name, open(file_path, "rb"), "image/jpeg"))
            for file_path in selected_files
        ]
        with self.client.post(
            "/predict-batch?model=baseline",
            files=files_payload,
            name="/predict-batch",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed: {response.status_code}")

