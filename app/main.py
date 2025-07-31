import os
import logging
from pydantic_settings import BaseSettings
import google.generativeai as genai

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import RunnablePassthrough

from app.schemas import FinalReport
from app.core.analyzer import (
    parse_resume_from_pdf,
    create_scoring_runnable,
    assemble_final_report
)

# Configuration Management
class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str
    LANGCHAIN_API_KEY: str
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    class Config:
        env_file = ".env"

settings = Settings()

# Configure Libraries
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App Initialization
app = FastAPI(
    title="Resume Analysis API",
    description="An API to parse and score a resume PDF using LangChain and generative AI.",
    version="1.0.0"
)


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


# Global LangChain Runnable Initialization
scoring_runnable = create_scoring_runnable()

full_chain = (
    RunnablePassthrough.assign(
       parsed_data=lambda x: parse_resume_from_pdf(x['pdf_bytes'], x['filename'])
    ).assign(
       scores=lambda x: scoring_runnable.invoke(x['parsed_data'])
    ) | assemble_final_report
)

# API Endpoint 
@app.post("/analyze-resume/", response_model=FinalReport, tags=["Resume Analysis"])
async def analyze_resume(file: UploadFile = File(..., description="The resume PDF file to be analyzed.")):
    """
    Upload a resume PDF to receive a comprehensive analysis and score.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    
    logger.info(f"\n--- ✅ Received request for file: {file.filename} ---\n")

    try:
        pdf_bytes = await file.read()
        chain_input = {"pdf_bytes": pdf_bytes, "filename": file.filename}
        final_json_output = await full_chain.ainvoke(chain_input)
        logger.info("\n\n--- ✅ ANALYSIS COMPLETE ---\n")
        return final_json_output

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Processing Error: {e}")
    except Exception as e:
        logger.error(f"\n--- ❌ A critical error occurred during chain execution: {e} ---")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
