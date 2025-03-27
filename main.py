from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
import httpx
import pathlib
from io import BytesIO

# Initialize the FastAPI app and GenAI client
app = FastAPI()
client = genai.Client()

# Define the summarization endpoint
@app.post("/summarize-pdf/")
async def summarize_pdf(file: UploadFile = File(...)):
    try:
        # Step 1: Save the uploaded PDF file to a temporary location
        pdf_content = await file.read()
        file_path = pathlib.Path("temp_file.pdf")
        file_path.write_bytes(pdf_content)

        # Step 2: Set up the prompt and call the GenAI API to summarize the PDF content
        prompt = "Summarize this document"
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_content,
                    mime_type='application/pdf',
                ),
                prompt
            ]
        )

        # Step 3: Return the summarized result as a JSON response
        return JSONResponse(content={"summary": response.text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
