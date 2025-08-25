from fastapi import FastAPI, UploadFile, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from .utils import load_model, get_classes, get_fci_links, preprocess_image
from carni_detect.config import InferenceConfig
import numpy as np


model = None
class_names = None
fci_links = None
config = InferenceConfig()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, fci_links
    model = load_model()
    class_names = get_classes()
    fci_links = get_fci_links()
    yield
    model = None
    class_names = None
    fci_links = None


app = FastAPI(title="Carni Detect API", lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile):
    """
    Endpoint for making predictions on uploaded images.

    Args:
        file (UploadFile): The uploaded image file.
    """
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(
                status_code=400, detail="Only JPEG or PNG files are supported."
            )

        image = await preprocess_image(file, config.INFERENCE_IMAGE_SIZE)
        predictions = model.predict(image)

        confidence = float(np.max(predictions))

        if confidence < config.INFERENCE_CONFIDENCE_THRESHOLD:
            return JSONResponse(content={"success": 0}, status_code=200)
        else:
            pred = np.argmax(predictions)
            return JSONResponse(
                content={
                    "success": 1,
                    "name": class_names[pred],
                    "link": fci_links[pred],
                },
                status_code=200,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
