import sys
import os
import time
from typing import Any, Dict, List, Optional

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json

app = FastAPI()

# Add CORS middleware to handle cross-origin requests from the UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Vite dev server default
        "http://localhost:5173",  # Vite dev server alternative
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Import callback router with error handling
try:
    from app.callbacks.endpoints import router as callbacks_router
    app.include_router(callbacks_router)
    CALLBACKS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import callback endpoints - {str(e)}")
    CALLBACKS_AVAILABLE = False

# Import LLM for direct chat functionality
try:
    from app.llm import LLM
    from app.schema import Message
    CHAT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import LLM - {str(e)}")
    CHAT_AVAILABLE = False

# Global LLM instance (will be initialized on first use)
llm_instance = None
# Store conversation history per conversation_id
conversation_history = {}
# Track task progress by conversation ID
task_progress = {}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/progress/{conversation_id}")
async def get_progress(conversation_id: str):
    """Get task progress for a specific conversation"""
    progress = task_progress.get(conversation_id, {
        "status": "idle",
        "message": "No active task",
        "progress": 0
    })
    return progress

def update_progress(conversation_id: str, status: str, message: str, progress: int = 0):
    """Update task progress for a conversation"""
    task_progress[conversation_id] = {
        "status": status,
        "message": message,
        "progress": progress,
        "timestamp": time.time()
    }

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    task_type: Optional[str] = None
    conversation_id: Optional[str] = None
    callback_config: Optional[Dict[str, Any]] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    global llm_instance, conversation_history, task_progress
    
    if not CHAT_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Chat functionality is not available due to missing dependencies"
        )
    
    try:
        # Get conversation ID
        conv_id = request.conversation_id or "default"
        task_type = request.task_type or 'general_chat'
        
        # Debug logging
        print(f"DEBUG: Received task_type: {task_type}")
        print(f"DEBUG: Request data: {request.dict()}")
        
        # Initialize conversation history if needed
        if conv_id not in conversation_history:
            conversation_history[conv_id] = []
        
        # Add user message to conversation history
        user_message = Message.user_message(request.message)
        conversation_history[conv_id].append(user_message)
        
        # Handle different task types with appropriate functionality
        if task_type == 'web_browsing':
            # Update progress
            update_progress(conv_id, "processing", "Initializing web browsing capabilities...", 10)
            
            # For now, provide enhanced web browsing guidance with LLM
            # TODO: Implement full Manus agent integration when browser setup is complete
            if llm_instance is None:
                config_name = "default"
                if request.model and request.model != "auto":
                    config_name = request.model.replace("/", "_").replace("-", "_")
                llm_instance = LLM(config_name=config_name)
            
            update_progress(conv_id, "processing", "Analyzing web research request...", 30)
            
            system_message = Message.system_message(
                f"""🌐 WEB BROWSING MODE ACTIVATED 🌐
                
                You are a specialized web research assistant. The user has requested: {request.message}
                
                IMPORTANT: Start your response with "🌐 WEB BROWSING TASK DETECTED" to clearly indicate this is different from regular chat.
                
                Since you don't have direct web browsing capabilities in this response, please:
                1. ✅ Acknowledge that this is a web browsing task (start with the indicator above)
                2. 🎯 Explain what specific websites or sources would be most relevant
                3. 🔍 Provide guidance on what information to look for
                4. 📝 Suggest specific search terms or strategies
                5. ⚡ Indicate that enhanced web browsing functionality is being implemented
                
                Be helpful and specific about the research approach. Format your response clearly with sections.
                Note: Full automated web browsing capabilities with real-time web access are being implemented.
                """
            )
            
            update_progress(conv_id, "processing", "Generating web research guidance...", 70)
            
            response_content = await llm_instance.ask(
                messages=conversation_history[conv_id],
                system_msgs=[system_message],
                stream=False,
                temperature=0.7
            )
            
            update_progress(conv_id, "completed", "Web research guidance completed", 100)
            
        elif task_type == 'code_generation':
            # Use enhanced LLM for code generation
            if llm_instance is None:
                config_name = "default"
                if request.model and request.model != "auto":
                    config_name = request.model.replace("/", "_").replace("-", "_")
                llm_instance = LLM(config_name=config_name)
            
            system_message = Message.system_message(
                "You are an expert software developer and coding assistant. Help users write, debug, and optimize code. Provide clear explanations, follow best practices, and include comments in your code examples. Focus on writing clean, efficient, and maintainable code."
            )
            
            response_content = await llm_instance.ask(
                messages=conversation_history[conv_id],
                system_msgs=[system_message],
                stream=False,
                temperature=0.3  # Lower temperature for more consistent code
            )
            
        elif task_type == 'data_analysis':
            # Use Manus agent with Python execution for data analysis
            from app.agent.manus import Manus
            
            manus = await Manus.create()
            
            data_prompt = f"""You are a data analysis expert with Python execution capabilities.
            The user has requested: {request.message}
            
            Please help them by:
            1. Understanding their data analysis needs
            2. Using Python code execution to analyze data, create visualizations, or perform statistical analysis
            3. Providing clear explanations of methodology and findings
            4. Offering actionable insights based on the analysis
            
            Use the python_execute tool as needed to perform calculations, create charts, or analyze data.
            """
            
            result = await manus.think()
            response_content = result.get('response', 'I encountered an issue during data analysis. Please try again.')
            
            await manus.cleanup()
            
        else:
            # Use basic LLM for general chat, reasoning, and creative writing
            if llm_instance is None:
                config_name = "default"
                if request.model and request.model != "auto":
                    config_name = request.model.replace("/", "_").replace("-", "_")
                llm_instance = LLM(config_name=config_name)
            
            system_prompts = {
                'general_chat': "You are a helpful AI assistant. Provide clear, concise, and helpful responses to user questions and requests. Engage in natural conversation and be friendly and informative.",
                'reasoning': "You are a logical reasoning expert. Help users think through complex problems step-by-step. Break down problems into smaller parts, analyze relationships, and provide clear, logical explanations for your conclusions. Focus on critical thinking and problem-solving.",
                'creative_writing': "You are a creative writing assistant. Help users with storytelling, creative content, poetry, and imaginative writing. Be creative, engaging, and help develop compelling narratives, characters, and ideas. Focus on creativity and literary quality."
            }
            
            system_message = Message.system_message(
                system_prompts.get(task_type, system_prompts['general_chat'])
            )
            
            response_content = await llm_instance.ask(
                messages=conversation_history[conv_id],
                system_msgs=[system_message],
                stream=False,
                temperature=0.7
            )
        
        # Add assistant response to conversation history
        assistant_message = Message.assistant_message(response_content)
        conversation_history[conv_id].append(assistant_message)
        
        # Keep conversation history manageable (last 20 messages)
        if len(conversation_history[conv_id]) > 20:
            conversation_history[conv_id] = conversation_history[conv_id][-20:]
        
        return {
            "response": response_content,
            "conversation_id": conv_id,
            "model": request.model or "auto",
            "task_type": task_type,
            "status": "completed"
        }
        
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat"""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get('type') == 'chat':
                # Create ChatRequest from WebSocket data
                chat_request = ChatRequest(
                    message=message_data.get('message', ''),
                    model=message_data.get('model'),
                    task_type=message_data.get('task_type'),
                    conversation_id=message_data.get('conversation_id'),
                    callback_config=message_data.get('callback_config')
                )
                
                try:
                    # Process the chat request (reuse the existing chat logic)
                    # For now, we'll process it the same way as HTTP and send the result
                    # In the future, this could be enhanced for true streaming
                    
                    # Get conversation ID
                    conv_id = chat_request.conversation_id or "default"
                    task_type = chat_request.task_type or 'general_chat'
                    
                    # Debug logging
                    print(f"WebSocket DEBUG: Received task_type: {task_type}")
                    
                    # Initialize conversation history if needed
                    if conv_id not in conversation_history:
                        conversation_history[conv_id] = []
                    
                    # Add user message to conversation history
                    user_message = Message.user_message(chat_request.message)
                    conversation_history[conv_id].append(user_message)
                    
                    # Handle different task types (same logic as HTTP endpoint)
                    if task_type == 'web_browsing':
                        # Update progress
                        update_progress(conv_id, "processing", "Initializing web browsing capabilities...", 10)
                        
                        # Send progress update via WebSocket
                        await websocket.send_text(json.dumps({
                            "type": "progress",
                            "status": "processing",
                            "message": "Initializing web browsing capabilities...",
                            "progress": 10
                        }))
                        
                        # For now, provide enhanced web browsing guidance with LLM
                        if llm_instance is None:
                            config_name = "default"
                            if chat_request.model and chat_request.model != "auto":
                                config_name = chat_request.model.replace("/", "_").replace("-", "_")
                            llm_instance = LLM(config_name=config_name)
                        
                        update_progress(conv_id, "processing", "Analyzing web research request...", 30)
                        await websocket.send_text(json.dumps({
                            "type": "progress",
                            "status": "processing",
                            "message": "Analyzing web research request...",
                            "progress": 30
                        }))
                        
                        system_message = Message.system_message(
                            f"""🌐 WEB BROWSING MODE ACTIVATED 🌐
                            
                            You are a specialized web research assistant. The user has requested: {chat_request.message}
                            
                            IMPORTANT: Start your response with "🌐 WEB BROWSING TASK DETECTED" to clearly indicate this is different from regular chat.
                            
                            Since you don't have direct web browsing capabilities in this response, please:
                            1. ✅ Acknowledge that this is a web browsing task (start with the indicator above)
                            2. 🎯 Explain what specific websites or sources would be most relevant
                            3. 🔍 Provide guidance on what information to look for
                            4. 📝 Suggest specific search terms or strategies
                            5. ⚡ Indicate that enhanced web browsing functionality is being implemented
                            
                            Be helpful and specific about the research approach. Format your response clearly with sections.
                            Note: Full automated web browsing capabilities with real-time web access are being implemented.
                            """
                        )
                        
                        update_progress(conv_id, "processing", "Generating web research guidance...", 70)
                        await websocket.send_text(json.dumps({
                            "type": "progress",
                            "status": "processing",
                            "message": "Generating web research guidance...",
                            "progress": 70
                        }))
                        
                        response_content = await llm_instance.ask(
                            messages=conversation_history[conv_id],
                            system_msgs=[system_message],
                            stream=False,
                            temperature=0.7
                        )
                        
                        update_progress(conv_id, "completed", "Web research guidance completed", 100)
                        
                    else:
                        # Handle other task types with basic LLM
                        if llm_instance is None:
                            config_name = "default"
                            if chat_request.model and chat_request.model != "auto":
                                config_name = chat_request.model.replace("/", "_").replace("-", "_")
                            llm_instance = LLM(config_name=config_name)
                        
                        system_prompts = {
                            'general_chat': "You are a helpful AI assistant. Provide clear, concise, and helpful responses to user questions and requests. Engage in natural conversation and be friendly and informative.",
                            'code_generation': "You are an expert software developer and coding assistant. Help users write, debug, and optimize code. Provide clear explanations, follow best practices, and include comments in your code examples. Focus on writing clean, efficient, and maintainable code.",
                            'reasoning': "You are a logical reasoning expert. Help users think through complex problems step-by-step. Break down problems into smaller parts, analyze relationships, and provide clear, logical explanations for your conclusions. Focus on critical thinking and problem-solving.",
                            'creative_writing': "You are a creative writing assistant. Help users with storytelling, creative content, poetry, and imaginative writing. Be creative, engaging, and help develop compelling narratives, characters, and ideas. Focus on creativity and literary quality."
                        }
                        
                        system_message = Message.system_message(
                            system_prompts.get(task_type, system_prompts['general_chat'])
                        )
                        
                        response_content = await llm_instance.ask(
                            messages=conversation_history[conv_id],
                            system_msgs=[system_message],
                            stream=False,
                            temperature=0.7
                        )
                    
                    # Add assistant response to conversation history
                    assistant_message = Message.assistant_message(response_content)
                    conversation_history[conv_id].append(assistant_message)
                    
                    # Keep conversation history manageable (last 20 messages)
                    if len(conversation_history[conv_id]) > 20:
                        conversation_history[conv_id] = conversation_history[conv_id][-20:]
                    
                    # Send response via WebSocket
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "response": response_content,
                        "conversation_id": conv_id,
                        "model": chat_request.model or "auto",
                        "task_type": task_type,
                        "status": "completed"
                    }))
                    
                except Exception as e:
                    print(f"WebSocket chat error: {str(e)}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": f"Chat error: {str(e)}"
                    }))
                    
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

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
