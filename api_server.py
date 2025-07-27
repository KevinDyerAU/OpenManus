import sys
import os
from typing import Any, Dict, List, Optional

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Import callback router with error handling
try:
    from app.callbacks.endpoints import router as callbacks_router
    app.include_router(callbacks_router)
    CALLBACKS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import callback endpoints - {str(e)}")
    CALLBACKS_AVAILABLE = False

# Import tools with error handling
try:
    from app.tool.create_chat_completion import CreateChatCompletion
    from app.tool.planning import PlanningTool
    CHAT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all tools - {str(e)}")
    CHAT_AVAILABLE = False

# Initialize tools with error handling
if CHAT_AVAILABLE:
    try:
        chat_tool = CreateChatCompletion()
        planning_tool = PlanningTool()
    except Exception as e:
        print(f"Warning: Could not initialize tools - {str(e)}")
        CHAT_AVAILABLE = False

@app.get("/health")
async def health():
    return {"status": "ok"}

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if not CHAT_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Chat functionality is not available due to missing dependencies"
        )
    try:
        result = await chat_tool.execute(response=request.prompt)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

class TaskRequest(BaseModel):
    command: str
    plan_id: Optional[str] = None
    title: Optional[str] = None
    steps: Optional[List[str]] = None

@app.post("/task")
async def task(request: TaskRequest):
    if not CHAT_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Task functionality is not available due to missing dependencies"
        )
    try:
        result = await planning_tool.execute(
            command=request.command,
            plan_id=request.plan_id,
            title=request.title,
            steps=request.steps
        )
        # Handle different result types
        if hasattr(result, 'output'):
            return {"result": result.output}
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
