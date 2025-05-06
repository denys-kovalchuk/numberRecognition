from fastapi import FastAPI, UploadFile
from fastapi.params import File

from ML_number_recognizer.predict_number_scikit_learn import main

app = FastAPI()


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str) -> dict[str, str]:
    return {"message": f"Hello {name}"}


@app.get("/test")
async def test() -> dict[str, str]:
    return {"message": "test"}


@app.post("/number")
async def predict_number(file: UploadFile = File(...)) -> int:
    predict = main(file)
    return predict
