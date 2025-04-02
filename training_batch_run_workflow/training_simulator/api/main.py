# main.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.wsgi import WSGIMiddleware
from training_batch_run_workflow.training_simulator.training_simulator.default import DEFAULT_PARAMETERS
from training_batch_run_workflow.training_simulator.training_simulator.batchrunner import parameter_sweep
from app2 import app as app_dash
import uvicorn
import json

app = FastAPI()

# Serve static files (e.g., CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def home():
    return {"message": "Welcome to our integrated FastAPI-Dash application"}

# Backend logic in FastAPI

@app.post("/simulate/")
async def process_data(data: dict):
    print(data)
    df = parameter_sweep(
        param_overrides=data, parameters=DEFAULT_PARAMETERS
    )
    ''' simulation results are outputted as dataframe with key columns being
    progressing_<coursenum>_count, hold_<coursenum>_count, left_raf_<coursenum>_count
    which map to progressing number of trainees on course, trainees on hold for course and course attrited trainee
    numbers data to be visualised on the front end. We also have training complete numbers for each pathway
    in this case 4 pathways training_pathway1_complete, training_pathway2_complete, training_pathway3_complete
    and training_pathway4_complete
    '''
   # print(df.columns)
   # print(parse_csv(df))
    df.set_index('Step')
   
    return df.to_json(orient='index')



app.mount("/dashboard/", WSGIMiddleware(app_dash.server))

# Start the FastAPI server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8055, reload=False, log_level="debug")