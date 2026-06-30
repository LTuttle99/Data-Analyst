import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request
import json

from analyzer import BookOfBusinessAnalyzer

app = FastAPI(title="Book of Business Intelligent Analyzer")

# Basic memory cache for demonstration simplicity
UPLOADED_FILE_CACHE = {}

# BASE_DIR finds the absolute root of your project folder wherever it runs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        analyzer = BookOfBusinessAnalyzer(contents, file.filename)
        schema = analyzer.infer_schema()
        
        # Cache analyzer instance under a fixed session placeholder
        UPLOADED_FILE_CACHE["current_session"] = analyzer
        
        return JSONResponse(content=schema)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/analyze")
async def analyze_data(request: Request):
    try:
        body = await request.json()
        mapping = body.get("mapping")
        
        analyzer = UPLOADED_FILE_CACHE.get("current_session")
        if not analyzer:
            raise HTTPException(status_code=400, detail="No uploaded file found in workspace context.")
            
        results = analyzer.run_analysis(mapping)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    # Construct a bulletproof absolute path to templates/index.html
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail=f"Template not found at structural path: {template_path}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
