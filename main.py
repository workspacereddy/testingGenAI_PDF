import pathlib
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from google.genai import types
from dotenv import load_dotenv
import httpx

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Retrieve the API key from the environment
API_KEY = os.getenv("GENAI_API_KEY")

# Check if API_KEY is available
if not API_KEY:
    raise ValueError("API_KEY is missing. Please make sure to set it in the .env file.")

# Endpoint to handle PDF file upload and summarization
@app.post("/summarize-pdf/")
async def summarize_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded PDF content temporarily
        pdf_content = await file.read()

        # Optionally, save to a local file (for logging purposes or other needs)
        temp_file_path = pathlib.Path("temp_uploaded_pdf.pdf")
        temp_file_path.write_bytes(pdf_content)

        # Prompt for summarization (you can adjust the prompt as needed)
        prompt = "Summarize this document"

        # Create an HTTPX client to send a request to Google GenAI API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://genai.googleapis.com/v1/models/gemini-1.5-flash:generateContent",
                json={
                    "model": "gemini-1.5-flash",  # Replace with your desired model
                    "contents": [
                        {
                            "data": pdf_content,
                            "mime_type": "application/pdf",
                        },
                        prompt,
                    ],
                },
                headers={"Authorization": f"Bearer {API_KEY}"},
            )

            # Check if the response was successful
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to summarize PDF.")

            # Extract the summarized content from the response
            summary = response.json().get("text", "")

            # Return the summarized text as a JSON response
            return JSONResponse(content={"summary": summary})

    except httpx.HTTPStatusError as http_error:
        # Return HTTP error message
        raise HTTPException(status_code=http_error.response.status_code, detail=str(http_error))
    except Exception as e:
        # Handle any other unexpected errors
        return JSONResponse(status_code=500, content={"error": str(e)})

