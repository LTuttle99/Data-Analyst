import os
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.requests import Request

from analyzer import BookOfBusinessAnalyzer

app = FastAPI(title="Book of Business Intelligent Analyzer")

UPLOADED_FILE_CACHE = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        analyzer = BookOfBusinessAnalyzer(contents, file.filename)
        schema = analyzer.infer_schema()
        
        UPLOADED_FILE_CACHE["current_session"] = analyzer
        return JSONResponse(content=schema)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)  # Appears in Render logs
        return JSONResponse(status_code=400, content={"error": f"{str(e)}\n\nTraceback:\n{error_trace}"})

@app.post("/api/analyze")
async def analyze_data(request: Request):
    try:
        body = await request.json()
        mapping = body.get("mapping")
        profit_center = body.get("profit_center", "ALL")
        projection_target = body.get("projection_target", "premium")
        
        analyzer = UPLOADED_FILE_CACHE.get("current_session")
        if not analyzer:
            raise HTTPException(status_code=400, detail="No active data file found in session memory.")
            
        results = analyzer.run_analysis(mapping, profit_center, projection_target)
        return JSONResponse(content=results)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)  # Appears in Render logs
        return JSONResponse(status_code=500, content={"error": f"{str(e)}\n\nTraceback:\n{error_trace}"})

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Template index.html missing from repository layout structure.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
