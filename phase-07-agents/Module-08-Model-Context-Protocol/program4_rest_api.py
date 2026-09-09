from fastapi import FastAPI, HTTPException


# Create the REST API application.
app = FastAPI(title="DataOps REST API")


# Simulate jobs stored in an existing system.
JOBS = {
    101: {
        "job_id": 101,
        "job_name": "daily_customer_etl",
        "status": "FAILED",
        "error": "ExecutorOutOfMemory",
    },
    102: {
        "job_id": 102,
        "job_name": "customer_dimension_load",
        "status": "RUNNING",
        "error": None,
    },
}


# Expose a REST endpoint for retrieving job status.
@app.get("/jobs/{job_id}")
def get_job(job_id: int):

    # Check whether the requested job exists.
    if job_id not in JOBS:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found",
        )

    # Return the job information as JSON.
    return JOBS[job_id]