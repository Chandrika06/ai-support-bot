import os
import csv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from huggingface_hub import InferenceApi

# 1️⃣ Load environment variables from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "google/flan-t5-base")

# 2️⃣ Initialize the Hugging Face Inference client
inference = InferenceApi(repo_id=HF_MODEL, token=HF_TOKEN)

app = FastAPI(title="HF Support Agent")

class QueryRequest(BaseModel):
    question: str

def load_faqs_local():
    faqs = []
    # Build the path to faq_data/faqs.csv (two levels up from hf_support/)
    path = os.path.join(
        os.path.dirname(__file__),
        "..",       # from hf_support/ to backend/
        "..",       # from backend/ to project root
        "faq_data",
        "faqs.csv"
    )
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                faqs.append(row)
        return faqs
    except Exception as e:
        raise RuntimeError(f"Could not read local faqs.csv: {e}")

def find_relevant_faqs(question: str, faqs: list, top_k: int = 3):
    q_lower = question.lower()
    ranked = []
    for item in faqs:
        faq_q = item["question"].lower()
        # Count overlap of words
        score = sum(word in q_lower for word in faq_q.split())
        if score > 0:
            ranked.append((score, item))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [item for (_, item) in ranked[:top_k]]

@app.post("/ask")
async def ask_support(req: QueryRequest):
    try:
        user_q = req.question.strip()
        if not user_q:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # 1️⃣ Load FAQs
        faqs = load_faqs_local()

        # 2️⃣ Find relevant FAQs
        relevant = find_relevant_faqs(user_q, faqs, top_k=3)
        context_snippets = "\n".join(
            [f"Q: {f['question']}\nA: {f['answer']}" for f in relevant]
        )

        # 3️⃣ Build the prompt
        prompt = (
            "You are a helpful customer support assistant.\n\n"
            "Here are some relevant FAQs:\n"
            f"{context_snippets}\n\n"
            f"User Question: {user_q}\n\n"
            "Answer:"
        )

        # 4️⃣ Call Hugging Face Inference API with raw_response to get plain text
        hf_response = inference(inputs=prompt, raw_response=True)
        # hf_response.content is bytes; decode to string
        answer = hf_response.content.decode("utf-8")

        return {"answer": answer, "relevant_faqs": relevant}

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
