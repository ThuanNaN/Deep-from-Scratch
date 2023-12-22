import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

app = FastAPI()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/", tags=["root"])
async def read_root() -> dict:
    return {"message": "Welcome to your todo list."}

@app.post('/generate')
def generate(prompt: str):
    client = httpclient.InferenceServerClient(url="localhost:8000")
    # Inputs
    text_obj = np.array([prompt], dtype="object")

    input_tensors = [
        httpclient.InferInput("TEXT", text_obj.shape, np_to_triton_dtype(text_obj.dtype)),
    ]
    input_tensors[0].set_data_from_numpy(text_obj)

    output = [
        httpclient.InferRequestedOutput("TEXT_OUT"),
    ]
    # Query
    query_response = client.infer(model_name="ensemble_model",
                                inputs=input_tensors,
                                outputs=output)
    response = query_response.as_numpy('TEXT_OUT')[0].decode("UTF-8")

    return { 'message': response}
