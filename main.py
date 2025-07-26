import argparse
import asyncio
import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.agent.manus import Manus
from app.logger import logger

app = FastAPI(title="OpenManus API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class ChatRequest(BaseModel):
    message: str
    callback_url: Optional[str] = None
    session_id: Optional[str] = None


class TaskRequest(BaseModel):
    task: str
    callback_url: Optional[str] = None
    task_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


# Response models
class ChatResponse(BaseModel):
    response: str
    session_id: str
    status: str = "completed"


class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    progress: Optional[int] = None


@app.get("/")
async def root():
    return {"message": "OpenManus API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/task", response_model=TaskResponse)
async def task_endpoint(request: TaskRequest):
    """
    Task endpoint with optional callback support
    """
    try:
        import uuid

        task_id = request.task_id or str(uuid.uuid4())

        # Start task processing asynchronously
        asyncio.create_task(
            process_task_async(
                task_id, request.task, request.parameters or {}, request.callback_url
            )
        )

        return TaskResponse(task_id=task_id, status="started", progress=0)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Manus agent with a prompt")
    parser.add_argument(
        "--prompt", type=str, required=False, help="Input prompt for the agent"
    )
    args = parser.parse_args()

    # Create and initialize Manus agent
    agent = await Manus.create()
    try:
        # Use command line prompt if provided, otherwise ask for input
        prompt = args.prompt if args.prompt else input("Enter your prompt: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        await agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")
    finally:
        # Ensure agent resources are cleaned up before exiting
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
