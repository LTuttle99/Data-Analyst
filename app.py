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
        print(error_trace)
        return JSONResponse(status_code=400, content={"error": f"{str(e)}\n\nTraceback:\n{error_trace}"})

@app.post("/api/profit-centers")
async def refresh_profit_centers(request: Request):
    try:
        body = await request.json()
        pc_col = body.get("profit_center_column")
        analyzer = UPLOADED_FILE_CACHE.get("current_session")
        if not analyzer:
            raise HTTPException(status_code=400, detail="No active data file found in session memory.")
        return JSONResponse(content={"profit_centers": analyzer.get_profit_centers(pc_col)})
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return JSONResponse(status_code=500, content={"error": f"{str(e)}\n\nTraceback:\n{error_trace}"})

@app.post("/api/agency-codes")
async def refresh_agency_codes(request: Request):
    try:
        body = await request.json()
        agency_col = body.get("agency_code_column")
        analyzer = UPLOADED_FILE_CACHE.get("current_session")
        if not analyzer:
            raise HTTPException(status_code=400, detail="No active data file found in session memory.")
        return JSONResponse(content={"agency_codes": analyzer.get_agency_codes(agency_col)})
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return JSONResponse(status_code=500, content={"error": f"{str(e)}\n\nTraceback:\n{error_trace}"})

@app.post("/api/date-range")
async def refresh_date_range(request: Request):
    try:
        body = await request.json()
        time_col = body.get("timeline_column")
        analyzer = UPLOADED_FILE_CACHE.get("current_session")
        if not analyzer:
            raise HTTPException(status_code=400, detail="No active data file found in session memory.")
        return JSONResponse(content=analyzer.get_date_range(time_col))
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
        return JSONResponse(status_code=500, content={"error": f"{str(e)}\n\nTraceback:\n{error_trace}"})

@app.post("/api/analyze")
async def analyze_data(request: Request):
    try:
        body = await request.json()
        mapping = body.get("mapping")
        profit_center = body.get("profit_center", "ALL")
        projection_target = body.get("projection_target", "premium")
        start_date = body.get("start_date")
        end_date = body.get("end_date")
        include_future_dates = bool(body.get("include_future_dates", False))
        selected_agency_codes = body.get("selected_agency_codes", []) or []
        business_view = body.get("business_view", "all")
        
        goal_raw = body.get("goal_value", 0)
        try:
            goal_value = float(goal_raw) if goal_raw not in (None, "", "null") else 0.0
        except (ValueError, TypeError):
            goal_value = 0.0
        
        analyzer = UPLOADED_FILE_CACHE.get("current_session")
        if not analyzer:
            raise HTTPException(status_code=400, detail="No active data file found in session memory.")
            
        results = analyzer.run_analysis(
            mapping, profit_center, projection_target,
            start_date, end_date, include_future_dates,
            selected_agency_codes, goal_value, business_view
        )
        return JSONResponse(content=results)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(error_trace)
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
