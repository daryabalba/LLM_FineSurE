import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PyPDF2 import PdfReader
from model_utils import load_model_and_tokenizer, generate_summary
from gigachat_integration import evaluate_summaries, extract_keyfacts
from config import BASE_DIR, MODEL_DIR
from start import train_model
import pandas as pd
from pathlib import Path


def init_application():
    fastapi_app = FastAPI(title="Text Summarization Service")
    if not MODEL_DIR.exists():
        print("Model not found. Starting training...")
        train_model()

    print("Loading model...")
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(MODEL_DIR)
    print("Model loaded successfully")
    return fastapi_app, finetuned_model, finetuned_tokenizer


app, model, tokenizer = init_application()
app.mount("/assets", StaticFiles(directory=Path(__file__).parent / "assets"), "assets")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


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
        summary = generate_summary(text, model, tokenizer)
        key_facts = extract_keyfacts(text)

        df_temp = pd.DataFrame({'article': [text], 'generated_summary': [summary]})
        summary_evaluation = evaluate_summaries(df_temp)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "original_text": text[:500] + "..." if len(text) > 500 else text,
                "summary": summary,
                "sum_evaluation": summary_evaluation.iloc[0].to_dict() if not summary_evaluation.empty else {},
                "key_facts": key_facts
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
