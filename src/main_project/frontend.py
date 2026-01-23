import json
import os
import subprocess
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Class names must match backend
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

def get_backend_url() -> str:
    """
    Get the URL of the backend service deployed on Cloud Run.
    Falls back to environment variable BACKEND if not found.
    Uses gcloud CLI instead of run_v2 client.
    """
    try:
        # List Cloud Run services using gcloud CLI
        result = subprocess.run(
            ["gcloud", "run", "services", "list", "--platform=managed", "--format=json", "--region=europe-west1"],
            capture_output=True,
            text=True,
            check=True
        )
        services = json.loads(result.stdout)
        for service in services:
            if service["metadata"]["name"] == "production-model":
                return service["status"]["url"]
    except subprocess.CalledProcessError as e:
        st.error(f"Failed to get Cloud Run services via gcloud: {e}")
    # Fallback to environment variable
    return os.environ.get("BACKEND", None)



def classify_image(image_bytes: bytes, backend: str) -> dict | None:
    """
    Send image bytes to backend /predict endpoint and return JSON result.
    """
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    try:
        response = requests.post(f"{backend}/predict?model=baseline", files=files, timeout=15)
        if response.status_code == 200:
            return response.json()
        st.warning(f"Prediction failed with status {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
    return None

def main() -> None:
    st.title("Chest X-Ray Classification")
    st.write("Upload a chest X-Ray image to get a prediction.")

    backend = get_backend_url()
    if not backend:
        st.error("Backend URL not found. Set BACKEND environment variable or deploy Cloud Run service.")
        return

    # FIXED: Make sure the file uploader is properly displayed
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Click here to browse for an image"
    )

    if uploaded_file is None:
        # Show a waiting message before user uploads
        st.info("👆 Please upload an image using the button above")
        return

    # Read image bytes
    image_bytes = uploaded_file.read()

    # Display the uploaded image
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Call backend
    with st.spinner("Analyzing X-Ray image..."):
        result = classify_image(image_bytes, backend)

    if result is None:
        st.write("Prediction could not be retrieved.")
        return

    # Display prediction
    prediction = result.get("predicted_class", "Unknown")
    confidence = result.get("confidence", 0)
    probabilities = result.get("probabilities", {})

    st.subheader("Prediction")
    st.write(f"**Class:** {prediction}")
    st.write(f"**Confidence:** {confidence:.2f}")

    # Bar chart for class probabilities
    df = pd.DataFrame({
        "Class": list(probabilities.keys()),
        "Probability": [probabilities[k] for k in probabilities.keys()]
    })
    df.set_index("Class", inplace=True)
    st.subheader("Class Probabilities")
    st.bar_chart(df)


if __name__ == "__main__":
    main()
