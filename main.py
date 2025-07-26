import asyncio
import os
import uuid
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="OpenManus API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    callback_url: Optional[str] = None
    session_id: Optional[str] = None


class TaskRequest(BaseModel):
    task: str
    callback_url: Optional[str] = None
    task_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


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
    return {"message": "OpenManus API is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "OpenManus"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Simple echo response for now - replace with actual OpenManus logic
        response = f"OpenManus processed: {request.message}"

        if request.callback_url:
            asyncio.create_task(
                send_callback(
                    request.callback_url,
                    {
                        "type": "chat_response",
                        "response": response,
                        "session_id": request.session_id or "default",
                    },
                )
            )

        return ChatResponse(
            response=response, session_id=request.session_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/task", response_model=TaskResponse)
async def task_endpoint(request: TaskRequest):
    try:
        task_id = request.task_id or str(uuid.uuid4())

        # Start task processing
        asyncio.create_task(
            process_task_async(
                task_id, request.task, request.parameters or {}, request.callback_url
            )
        )

        return TaskResponse(task_id=task_id, status="started", progress=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    return TaskResponse(
        task_id=task_id,
        status="completed",
        result=f"Task {task_id} completed successfully",
        progress=100,
    )


async def process_task_async(
    task_id: str,
    task: str,
    parameters: Dict[str, Any],
    callback_url: Optional[str] = None,
):
    try:
        # Simulate task processing
        await asyncio.sleep(2)
        result = f"Task completed: {task}"

        if callback_url:
            await send_callback(
                callback_url,
                {
                    "type": "task_completed",
                    "task_id": task_id,
                    "result": result,
                    "status": "completed",
                },
            )
    except Exception as e:
        if callback_url:
            await send_callback(
                callback_url,
                {
                    "type": "task_failed",
                    "task_id": task_id,
                    "error": str(e),
                    "status": "failed",
                },
            )


async def send_callback(callback_url: str, data: Dict[str, Any]):
    async with httpx.AsyncClient() as client:
        try:
            await client.post(callback_url, json=data, timeout=30)
        except Exception as e:
            print(f"Callback failed: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
