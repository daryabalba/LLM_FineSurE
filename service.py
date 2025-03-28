import os
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
from pathlib import Path
from PyPDF2 import PdfReader
import nltk
from main import generate_summary
import pandas as pd
from gigachat_integration import evaluate_summaries, extract_keyfacts


# Инициализация приложения
app = FastAPI(title="Text Summarization Service")

# Настройка путей
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Скачиваем необходимые данные для nltk
nltk.download('punkt')


# Функции для работы с файлами
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'txt'}


async def extract_text_from_file(file: UploadFile) -> str:
    content = await file.read()

    if file.filename.endswith('.pdf'):
        temp_path = f"temp_{file.filename}"
        with open(temp_path, 'wb') as f:
            f.write(content)

        text = ""
        with open(temp_path, 'rb') as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()

        os.remove(temp_path)
        return text
    else:
        return content.decode('utf-8')


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Invalid file type. Only PDF and TXT are allowed."}
        )

    try:
        text = await extract_text_from_file(file)

        summary = generate_summary(text)

        df_temp = pd.DataFrame({'article': text, 'generated_summary': summary})
        # Оценка качества


        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "original_text": text[:500] + "..." if len(text) > 500 else text,
                "sumy_summary": sumy_summary,
                "transformer_summary": transformer_summary,
                "sumy_evaluation": sumy_evaluation,
                "transformer_evaluation": transformer_evaluation
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"An error occurred: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)