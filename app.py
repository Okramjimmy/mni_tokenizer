"""
FastAPI Service for Meitei Mayek Sentence Splitter.

Usage:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from meitei_tokenizer import MeiteiTokenizer
import os
from typing import List

app = FastAPI(
    title="Meitei Mayek Tokenizer API",
    description="API for splitting Meitei Mayek text into sentences using a context-aware neural pipeline.",
    version="1.0.0"
)

# Global model variable
nlp = None

@app.on_event("startup")
def load_model():
    """Load the model on startup to avoid high latency per request."""
    global nlp
    try:
        model_path = "./output/model-best"
        spm_path = "meitei_tokenizer.model"
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Please train the model first.")
            return
            
        print("Loading spaCy model...")
        nlp = spacy.load(model_path)
        print("Loading SentencePiece tokenizer...")
        nlp.tokenizer = MeiteiTokenizer(spm_path, nlp.vocab)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

class SplitRequest(BaseModel):
    text: str

class SplitResponse(BaseModel):
    sentences: List[str]

@app.post("/split", response_model=SplitResponse)
async def split_sentences(request: SplitRequest):
    """
    Split input text into sentences.
    """
    global nlp
    if nlp is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    if not request.text:
        return {"sentences": []}
    
    doc = nlp(request.text)
    sentences = [sent.text for sent in doc.sents]
    
    return {"sentences": sentences}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": nlp is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
