import requests
import pandas as pd
import json
from tqdm import tqdm
import time
from gigachat import GigaChat
import re
from typing import List


GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
CLIENT_ID = "4e69c9e9-d948-4def-a13a-5611198e1cda"
CLIENT_SECRET = "NGU2OWM5ZTktZDk0OC00ZGVmLWExM2EtNTYxMTE5OGUxY2RhOmZlNWVmZTBkLTcyN2EtNDFjZS1iZGJlLThiYzBjN2YxMGM2NQ=="

with GigaChat(credentials=CLIENT_SECRET, ca_bundle_file="russian_trusted_root_ca.cer") as giga:
    access_token = giga.get_token()


def extract_json_from_response(response: str):
    """

    Извлекает JSON из строки ответа,
    даже если есть лишний текст

    """
    try:
        # ищем JSON с помощью регулярного выражения
        json_match = re.search(r'```json\n?({.*?}|\[.*?\])\n?```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Если нет блока с ```json, ищем чистый JSON
        json_match = re.search(r'(\{.*?\}|\[.*?\])', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Если ничего не найдено, вернем None
        return None
    except json.JSONDecodeError:
        return None


def extract_keyfacts(article: str) -> List[str]: # функция для извлечения ключевых фактов
    prompt = f"""Extract 3-5 key facts from this article. Return ONLY a JSON array with facts:

    Article: {article}

    Output format must be EXACTLY like this:
    ["fact 1", "fact 2", "fact 3"]

    Important: Do not include any additional text, comments or explanations outside the JSON array."""

    response = query_gigachat(prompt)
    try:
        return json.loads(response)

    except:
        return []


def evaluate_faithfulness(article: str, summary: str) -> float: # функция для оценки фактической точности
    prompt = f"""Analyze factual consistency between article and summary. Return ONLY a JSON file with this exact structure:
    {{
        "analysis": [
            {{
                "sentence": "exact sentence from summary",
                "label": "correct|incorrect|hallucination"
            }}
        ]
    }}

    Rules:
    1. "correct" - fully supported by article
    2. "incorrect" - contradicts article
    3. "hallucination" - cannot be verified

    Article: {article[:3000]}
    Summary: {summary[:1000]}

    Important: Return ONLY the JSON object with no additional text or explanations."""

    response = query_gigachat(prompt)
    try:
        result = json.loads(response)
        if "analysis" not in result:
            return 0.0
        correct = sum(1 for item in result["analysis"] if item["label"] == "correct")
        return correct / len(result["analysis"]) if result["analysis"] else 0.0
    except:
        return 0.0


def evaluate_coverage(article: str, summary: str) -> float:
    keyfacts = extract_keyfacts(article)
    if not keyfacts:
        return 0.0

    prompt = f"""Check if these key facts from the article appear in the summary. Return ONLY a JSON file with this exact structure:
    {{
        "coverage": [
            {{
                "fact": "exact fact text",
                "status": "covered|missing"
            }}
        ]
    }}

    Key Facts: {json.dumps(keyfacts, ensure_ascii=False)}
    Summary: {summary[:1000]}

    Important: Return ONLY the JSON object with no additional text or explanations."""

    response = query_gigachat(prompt)
    try:
        result = json.loads(response)
        if "coverage" not in result:
            return 0.0
        covered = sum(1 for item in result["coverage"] if item["status"] == "covered")
        return covered / len(result["coverage"]) if result["coverage"] else 0.0
    except Exception as e:
        print(f"Coverage evaluation error: {e}")
        return 0.0


def evaluate_summaries(df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    if sample_size:
        df = df.sample(sample_size)

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        article = str(row["article"])
        generated = str(row["generated_summary"])

        gen_faith = evaluate_faithfulness(article, generated)
        gen_cov = evaluate_coverage(article, generated)

        results.append({
            "article_id": str(row.get("id", "")),
            "generated_faithfulness": gen_faith,
            "generated_coverage": gen_cov,
        })

        time.sleep(1)

    return pd.DataFrame(results)


def query_gigachat(prompt: str, temperature: float = 0.3) -> str:
    payload = {
        "model": "GigaChat",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that always responds with VALID JSON ONLY, without any additional text or explanations."
            },
            {
                "role": "user",
                "content": prompt + "\n\nImportant: Return ONLY a valid JSON response. Do not include any additional text, comments or markdown formatting outside the JSON structure."
            }
        ],
        "temperature": temperature,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"}
    }

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token.access_token}'
    }

    try:
        response = requests.post(
            GIGACHAT_API_URL,
            headers=headers,
            json=payload,
            verify='/content/russian_trusted_root_ca.cer'
        )
        response.raise_for_status()

        try:
            return response.json()["choices"][0]["message"]["content"]
        except:
            json_data = extract_json_from_response(response.text)
            if json_data is not None:
                return json.dumps(json_data)
            raise ValueError("No valid JSON found in response")

    except Exception as e:
        print(f"API Error: {e}")
        return ""


def main():
    url = "https://gigachat.devices.sberbank.ru/api/v1/models"
    payload={}
    headers = {
      'Accept': 'application/json',
      'Authorization': f'Bearer {access_token.access_token}'
    }

    response = requests.request("GET", url, headers=headers, data=payload, verify='/content/russian_trusted_root_ca.cer') # эксплицитно добавила сертификат, потому что он его иначе не видел, а если использовать False, то он плохо работает

    if response.text:
        print("Response obtained, everything's fine")
    else:
        return



