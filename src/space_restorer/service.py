from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .cascade import load_models, cascade
import torch

model_gap = None
model_space = None
tokenizer_gap = None
tokenizer_space = None
device = "cpu"

cascade_config = {
    "max_tries": 5,
    "min_tries": 1,
    "spaces": 0.2,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Входной и выходной контекст сервиса"""
    global model_gap, model_space, tokenizer_gap, tokenizer_space
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {device}")
    model_space, model_gap, tokenizer_space, tokenizer_gap = load_models(device=device)
    yield


app = FastAPI(
    title="Space Restorer",
    description="Сервис для восстановления упущенных пробелов",
    lifespan=lifespan,
)


class RestorationRequest(BaseModel):
    text: str = Field(
        ..., description="Текст, в котором необходимо восстановить пробелы."
    )


class RestorationResponse(BaseModel):
    text: str = Field(..., description="Восстановленный текст.")


@app.post("/restore", response_model=RestorationResponse)
async def restore_text(request: RestorationRequest):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        text: исходный текст

    Returns:
        RestorationResponse: результат обработки текста
    """
    global model_gap, model_space, tokenizer_gap, tokenizer_space

    try:
        result = cascade(
            model_gap=model_gap,
            model_space=model_space,
            tokenizer_gap=tokenizer_gap,
            tokenizer_space=tokenizer_space,
            text=request.text,
            max_tries=cascade_config["max_tries"],
            min_tries=cascade_config["min_tries"],
            spaces=cascade_config["spaces"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Обработка текста неудалась: {e}")

    return RestorationResponse(text=result)
