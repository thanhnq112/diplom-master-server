from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from recognize import recognize
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))


@app.get("/")
def read_root():
    return {"message": "Hello from the other side!"}


@app.post("/api/recognize")
async def main_process(file_upload: UploadFile):
    image = load_image_into_numpy_array(await file_upload.read())

    response = recognize(image)
    print(response)

    return response
