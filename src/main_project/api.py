"""FastAPI application for Chest X-Ray classification predictions."""

from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from main_project.model import get_model

# Initialize FastAPI app
app = FastAPI(
    title="Chest X-Ray Classification API",
    description="API for predicting chest X-Ray classifications",
    version="1.0.0",
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Get absolute paths - api.py is in src/main_project/
SCRIPT_DIR = Path(__file__).resolve().parent  # /path/to/src/main_project
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # /path/to/MLOPS-PROJECT
MODELS_DIR = PROJECT_ROOT/ "models"

print(f"\n{'='*80}")
print(f"Script location: {SCRIPT_DIR}")
print(f"Project root: {PROJECT_ROOT}")
print(f"Models directory: {MODELS_DIR}")
print(f"Models dir exists: {MODELS_DIR.exists()}")

if MODELS_DIR.exists():
    model_files = list(MODELS_DIR.glob("*.pth"))
    print(f"Found model files: {model_files}")
else:
    print(f"WARNING: Models directory does not exist!")
    print(f"Looking in: {MODELS_DIR}")
print(f"{'='*80}\n")

# Load configuration
config_path = PROJECT_ROOT / "configs"
cfg = OmegaConf.load(config_path / "config.yaml")

# Available models from model.py
AVAILABLE_MODELS = ["baseline", "alexnet", "vgg16"]
MODEL_PATHS = {
    "baseline": MODELS_DIR / "baseline_best.pth",
    "alexnet": MODELS_DIR / "alexnet_best.pth",
    "vgg16": MODELS_DIR / "vgg16_best.pth",
}

NUM_CLASSES = cfg.model.num_classes if "model" in cfg else 2
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Image preprocessing (matching your data.py preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Global model cache
loaded_models = {}


def get_model_path(model_name: str) -> Path:
    """Get model path."""
    return MODEL_PATHS.get(model_name)


def preprocess_image(image_data: bytes) -> torch.Tensor:
    """
    Preprocess uploaded image using same pipeline as training data.
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        Preprocessed image tensor ready for model
    """
    try:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")


def load_model(model_name: str) -> torch.nn.Module:
    """
    Load or retrieve cached model.
    
    Args:
        model_name: Name of the model (baseline, alexnet, vgg16)
    
    Returns:
        Loaded model in eval mode
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model must be one of {AVAILABLE_MODELS}")
    
    # Return cached model if already loaded
    if model_name in loaded_models:
        print(f"✓ Using cached {model_name} model")
        return loaded_models[model_name]
    
    try:
        model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
        model_path = get_model_path(model_name)
        
        print(f"Loading model: {model_name}")
        print(f"Model path: {model_path}")
        print(f"Path exists: {model_path.exists()}")
        
        if not model_path.exists():
            available_files = list(MODELS_DIR.glob("*")) if MODELS_DIR.exists() else []
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Available files in {MODELS_DIR}: {available_files}"
            )
        
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model = model.to(device)
        model.eval()
        
        # Cache the model
        loaded_models[model_name] = model
        print(f"✓ Successfully loaded {model_name} from {model_path}")
        
        return model
    except FileNotFoundError as e:
        print(f"✗ File error: {e}")
        raise
    except Exception as e:
        print(f"✗ Error loading {model_name}: {e}")
        raise


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    model_used: str
    predicted_class: str
    confidence: float
    probabilities: dict
    success: bool


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Chest X-Ray Classification API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "predict_batch": "/predict-batch",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "available_models": AVAILABLE_MODELS,
        "cached_models": list(loaded_models.keys()),
        "device": device,
        "models_directory": str(MODELS_DIR),
        "models_exist": MODELS_DIR.exists(),
    }


@app.get("/models")
async def list_models():
    """List available models."""
    models_info = {}
    for model_name in AVAILABLE_MODELS:
        model_path = get_model_path(model_name)
        models_info[model_name] = {
            "path": str(model_path),
            "exists": model_path.exists() if model_path else False,
            "loaded": model_name in loaded_models,
        }
    
    return {
        "available_models": AVAILABLE_MODELS,
        "models": models_info,
        "loaded_models": list(loaded_models.keys()),
        "models_directory": str(MODELS_DIR),
        "directory_exists": MODELS_DIR.exists(),
    }


@app.get("/info")
async def model_info(model: str = Query("baseline", description="Model name")):
    """
    Get model information.
    
    Args:
        model: Model name (baseline, alexnet, vgg16)
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model. Choose from: {AVAILABLE_MODELS}"
        )
    
    model_path = get_model_path(model)
    
    return {
        "model_name": model,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "device": device,
        "model_path": str(model_path),
        "file_exists": model_path.exists() if model_path else False,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Query("baseline", description="Model to use (baseline, alexnet, vgg16)")
) -> PredictionResponse:
    """
    Predict classification for uploaded chest X-Ray image.
    
    Args:
        file: Uploaded image file (JPG, PNG, JPEG)
        model: Model name (baseline, alexnet, vgg16)
    
    Returns:
        PredictionResponse with predicted class and confidence
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {AVAILABLE_MODELS}"
        )
    
    try:
        loaded_model = load_model(model)
        image_data = await file.read()
        image_tensor = preprocess_image(image_data).to(device)
        
        with torch.no_grad():
            outputs = loaded_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
            predicted_idx = probabilities.argmax().item()
            confidence = probabilities[predicted_idx].item()
        
        prob_dict = {CLASS_NAMES[i]: float(probabilities[i].item()) for i in range(NUM_CLASSES)}
        
        return PredictionResponse(
            model_used=model,
            predicted_class=CLASS_NAMES[predicted_idx],
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            success=True,
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    model: str = Query("baseline", description="Model name")
):
    """Predict classifications for multiple images."""
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {AVAILABLE_MODELS}"
        )
    
    results = []
    for file in files:
        result = await predict(file, model=model)
        results.append(result.model_dump())
    
    return {"model_used": model, "predictions": results, "count": len(results)}