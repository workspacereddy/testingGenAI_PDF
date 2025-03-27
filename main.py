from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Dict
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

class ChatMessage(BaseModel):
    message: str

class HealthData(BaseModel):
    bloodPressure: str
    bloodSugar: str
    cholesterol: str
    heartRate: str
    temperature: str

@app.get("/")
@app.head("/")
async def read_root():
    return {"message": "API is working!"}

# Handling OPTIONS request explicitly for CORS pre-flight
@app.options("/api/chat/") 
async def handle_options_chat():
    return JSONResponse(content={}, status_code=200)

@app.options("/api/predict/") 
async def handle_options_predict():
    return JSONResponse(content={}, status_code=200)

# Extract text from PDF
def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(docx_file: BytesIO) -> str:
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Extract text from TXT
def extract_text_from_txt(txt_file: BytesIO) -> str:
    return txt_file.read().decode("utf-8")

# Endpoint to process medical document
@app.post("/api/process_document")
async def process_document(file: UploadFile = File(...)) -> Dict[str, str]:
    # Read the uploaded file
    try:
        content = await file.read()
        file_extension = file.filename.split(".")[-1].lower()

        # Extract text based on file type
        if file_extension == "pdf":
            text = extract_text_from_pdf(BytesIO(content))
        elif file_extension == "docx":
            text = extract_text_from_docx(BytesIO(content))
        elif file_extension == "txt":
            text = extract_text_from_txt(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF, DOCX, and TXT are allowed.")

        # Use the extracted text to generate a response using Google AI
        prompt = f"""
        You are a medical AI assistant. Analyze the following medical document and provide a summary:
        {text}
        Always include a disclaimer that this is not professional medical advice.
        """

        try:
            response = model.generate_content(prompt)
            return {"summary": response.text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in processing the request: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
