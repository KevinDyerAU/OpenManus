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
        # Import LLM at the beginning to ensure it's available for all task types
        from app.llm import LLM
        
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
        if task_type == 'manus_agent':
            # Use the full Manus agent (main.py functionality)
            update_progress(conv_id, "processing", "Initializing Manus agent...", 10)
            
            try:
                # Import the Manus agent and LLM
                from app.agent.manus import Manus
                from app.llm import LLM
                
                update_progress(conv_id, "processing", "Creating agent instance...", 20)
                
                # Create LLM instance with selected model
                config_name = "default"
                if request.model and request.model != "auto":
                    # Handle different model formats
                    if request.model.startswith("openrouter/"):
                        config_name = "openrouter"
                    elif "/" in request.model:
                        # Format: provider/model -> provider_model
                        config_name = request.model.replace("/", "_").replace("-", "_")
                    else:
                        config_name = request.model
                
                print(f"DEBUG: Creating Manus agent with LLM config: {config_name} for model: {request.model}")
                custom_llm = LLM(config_name=config_name)
                
                # Create Manus agent instance with custom LLM
                agent = await Manus.create(llm=custom_llm)
                
                update_progress(conv_id, "processing", "Processing your request...", 40)
                
                # Execute the agent with the user's prompt
                # Set up real-time logging capture for HTTP mode
                import io
                import logging
                from contextlib import redirect_stdout, redirect_stderr
                
                # Create a custom sink for Loguru to capture agent thoughts for HTTP
                captured_logs = []
                
                def http_sink(message):
                    try:
                        # Extract the log message
                        log_text = message.record["message"]
                        level = message.record["level"].name
                        
                        # Format and store the message
                        formatted_message = f"🤖 [{level}] {log_text}"
                        captured_logs.append(formatted_message)
                        print(f"DEBUG: Captured agent thought for HTTP: {formatted_message}")
                        
                    except Exception as e:
                        print(f"Error capturing log for HTTP: {e}")
                
                # Add the custom sink to Loguru logger
                from app.logger import logger
                sink_id = logger.add(http_sink, level="INFO", format="{message}")
                
                # Capture stdout/stderr as well
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                
                try:
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        await agent.run(request.message)
                    
                    # Get captured output
                    agent_output = stdout_capture.getvalue()
                    agent_errors = stderr_capture.getvalue()
                    
                finally:
                    # Remove the custom Loguru sink
                    logger.remove(sink_id)
                
                update_progress(conv_id, "processing", "Finalizing results...", 90)
                
                # Format the response - include captured agent thoughts
                response_parts = []
                
                # Add agent thoughts if captured
                if captured_logs:
                    response_parts.append("🧠 **Agent Thoughts:**\n" + "\n".join(captured_logs))
                
                # Add main output
                if agent_output.strip():
                    response_parts.append(f"🤖 **Manus Agent Results:**\n{agent_output.strip()}")
                elif agent_errors.strip():
                    response_parts.append(f"🤖 **Manus Agent Output:**\n{agent_errors.strip()}")
                else:
                    response_parts.append(f"🤖 **Manus Agent:**\nTask completed successfully. The agent processed your request: '{request.message}'")
                
                response_content = "\n\n---\n\n".join(response_parts)
                
                # Clean up agent resources
                await agent.cleanup()
                    
            except ImportError as e:
                response_content = f"🤖 Manus agent not available: {str(e)}. Please ensure all dependencies are installed."
                update_progress(conv_id, "error", "Manus agent unavailable", 100)
            except Exception as e:
                response_content = f"🤖 Manus agent error: {str(e)}. Please try again or contact support."
                update_progress(conv_id, "error", "Manus agent failed", 100)
            
            update_progress(conv_id, "completed", "Manus agent completed", 100)
            
        elif task_type == 'web_browsing':
            # Update progress
            update_progress(conv_id, "processing", "Initializing web browsing agent...", 10)
            
            try:
                # Import browser tool directly (Playwright-based)
                from app.tool.browser_use_tool import BrowserUseTool
                import asyncio
                import re
                
                update_progress(conv_id, "processing", "Setting up browser...", 30)
                
                # Create browser tool instance
                browser_tool = BrowserUseTool()
                
                update_progress(conv_id, "processing", "Navigating to website...", 50)
                
                # Extract URL from user message if present
                url_match = re.search(r'https?://[^\s]+', request.message)
                if url_match:
                    url = url_match.group(0)
                    
                    # Navigate to the URL
                    nav_result = await browser_tool.execute(
                        action="go_to_url",
                        url=url
                    )
                    
                    update_progress(conv_id, "processing", "Extracting content...", 70)
                    
                    # Extract content based on user request
                    extract_result = await browser_tool.execute(
                        action="extract_content",
                        goal=request.message
                    )
                    
                    response_content = f"🌐 **Web Browsing Results for {url}**\n\n{extract_result.output}"
                    
                else:
                    # If no URL found, perform web search
                    search_result = await browser_tool.execute(
                        action="web_search",
                        query=request.message
                    )
                    
                    update_progress(conv_id, "processing", "Analyzing search results...", 70)
                    
                    # Extract content from search results
                    extract_result = await browser_tool.execute(
                        action="extract_content",
                        goal=f"Find information about: {request.message}"
                    )
                    
                    response_content = f"🌐 **Web Search Results for '{request.message}'**\n\n{extract_result.output}"
                
                update_progress(conv_id, "completed", "Web browsing completed", 100)
                
            except asyncio.TimeoutError:
                response_content = "🌐 Web browsing task timed out after 5 minutes. Please try a simpler request or break it into smaller parts."
                update_progress(conv_id, "completed", "Web browsing timed out", 100)
            except ImportError as e:
                response_content = f"🌐 Web browsing agent not available: {str(e)}. Please ensure all dependencies are installed."
                update_progress(conv_id, "error", "Web browsing agent unavailable", 100)
            except Exception as e:
                response_content = f"🌐 Web browsing error: {str(e)}. Please try again or contact support."
                update_progress(conv_id, "error", "Web browsing failed", 100)
            
        elif task_type == 'code_generation':
            # Use enhanced LLM for code generation
            # Always create new LLM instance to support model switching
            config_name = "default"
            if request.model and request.model != "auto":
                # Handle different model formats
                if request.model.startswith("openrouter/"):
                    config_name = "openrouter"
                elif "/" in request.model:
                    # Format: provider/model -> provider_model
                    config_name = request.model.replace("/", "_").replace("-", "_")
                else:
                    config_name = request.model
            
            print(f"DEBUG: Using LLM config: {config_name} for model: {request.model}")
            current_llm = LLM(config_name=config_name)
            
            system_message = Message.system_message(
                "You are an expert software developer and coding assistant. Help users write, debug, and optimize code. Provide clear explanations, follow best practices, and include comments in your code examples. Focus on writing clean, efficient, and maintainable code."
            )
            
            response_content = await current_llm.ask(
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
            # Always create new LLM instance to support model switching
            config_name = "default"
            if request.model and request.model != "auto":
                # Handle different model formats
                if request.model.startswith("openrouter/"):
                    config_name = "openrouter"
                elif "/" in request.model:
                    # Format: provider/model -> provider_model
                    config_name = request.model.replace("/", "_").replace("-", "_")
                else:
                    config_name = request.model
            
            print(f"DEBUG: Using LLM config: {config_name} for model: {request.model}")
            current_llm = LLM(config_name=config_name)
            
            system_prompts = {
                'general_chat': "You are a helpful AI assistant. Provide clear, concise, and helpful responses to user questions and requests. Engage in natural conversation and be friendly and informative.",
                'reasoning': "You are a logical reasoning expert. Help users think through complex problems step-by-step. Break down problems into smaller parts, analyze relationships, and provide clear, logical explanations for your conclusions. Focus on critical thinking and problem-solving.",
                'creative_writing': "You are a creative writing assistant. Help users with storytelling, creative content, poetry, and imaginative writing. Be creative, engaging, and help develop compelling narratives, characters, and ideas. Focus on creativity and literary quality."
            }
            
            system_message = Message.system_message(
                system_prompts.get(task_type, system_prompts['general_chat'])
            )
            
            response_content = await current_llm.ask(
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
    global llm_instance  # Add global declaration
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
                    if task_type == 'manus_agent':
                        # Use the full Manus agent (main.py functionality) with progress updates
                        update_progress(conv_id, "processing", "Initializing Manus agent...", 10)
                        
                        # Send progress update via WebSocket
                        progress_msg = {
                            "type": "progress",
                            "status": "processing",
                            "message": "Initializing Manus agent...",
                            "progress": 10
                        }
                        await websocket.send_text(json.dumps(progress_msg))
                        print(f"DEBUG: Sent progress update via WebSocket: {progress_msg}")
                        
                        try:
                            # Import the Manus agent and LLM
                            from app.agent.manus import Manus
                            from app.llm import LLM
                            
                            update_progress(conv_id, "processing", "Creating agent instance...", 20)
                            
                            # Send progress update via WebSocket
                            progress_msg = {
                                "type": "progress",
                                "status": "processing",
                                "message": "Creating agent instance...",
                                "progress": 20
                            }
                            await websocket.send_text(json.dumps(progress_msg))
                            print(f"DEBUG: Sent progress update via WebSocket: {progress_msg}")
                            
                            # Create LLM instance with selected model
                            config_name = "default"
                            if chat_request.model and chat_request.model != "auto":
                                # Handle different model formats
                                if chat_request.model.startswith("openrouter/"):
                                    config_name = "openrouter"
                                elif "/" in chat_request.model:
                                    # Format: provider/model -> provider_model
                                    config_name = chat_request.model.replace("/", "_").replace("-", "_")
                                else:
                                    config_name = chat_request.model
                            
                            print(f"DEBUG: Creating WebSocket Manus agent with LLM config: {config_name} for model: {chat_request.model}")
                            custom_llm = LLM(config_name=config_name)
                            
                            # Create Manus agent instance with custom LLM
                            agent = await Manus.create(llm=custom_llm)
                            
                            update_progress(conv_id, "processing", "Processing your request...", 40)
                            
                            # Send progress update via WebSocket
                            progress_msg = {
                                "type": "progress",
                                "status": "processing",
                                "message": "Processing your request...",
                                "progress": 40
                            }
                            await websocket.send_text(json.dumps(progress_msg))
                            print(f"DEBUG: Sent progress update via WebSocket: {progress_msg}")
                            
                            # Execute the agent with the user's prompt
                            # Set up real-time logging capture and streaming
                            import io
                            import sys
                            import logging
                            from contextlib import redirect_stdout, redirect_stderr
                            
                            # Create a custom sink for Loguru to capture agent thoughts
                            captured_logs = []
                            
                            def websocket_sink(message):
                                try:
                                    # Extract the log message and level
                                    log_text = message.record["message"]
                                    level = message.record["level"].name
                                    
                                    # Only capture INFO and higher level messages to avoid spam
                                    if message.record["level"].no < 20:  # Below INFO level
                                        return
                                    
                                    # Format the message with emoji prefix for UI recognition
                                    formatted_message = f"🤖 [{level}] {log_text}"
                                    captured_logs.append(formatted_message)
                                    
                                    # Create progress message for UI
                                    progress_msg = {
                                        "type": "progress",
                                        "status": "processing",
                                        "message": formatted_message,
                                        "progress": 50  # Keep progress at 50% during execution
                                    }
                                    
                                    # Send WebSocket message via thread to avoid event loop conflicts
                                    try:
                                        import asyncio
                                        import threading
                                        import json
                                        
                                        def send_websocket_message():
                                            try:
                                                # Create a new event loop for this thread
                                                loop = asyncio.new_event_loop()
                                                asyncio.set_event_loop(loop)
                                                
                                                # Send the WebSocket message
                                                loop.run_until_complete(
                                                    websocket.send_text(json.dumps(progress_msg))
                                                )
                                                
                                            except Exception:
                                                pass  # Silently handle WebSocket send errors
                                            finally:
                                                try:
                                                    loop.close()
                                                except:
                                                    pass
                                        
                                        # Start the sending thread
                                        thread = threading.Thread(target=send_websocket_message)
                                        thread.daemon = True
                                        thread.start()
                                        
                                    except Exception:
                                        pass  # Silently handle any WebSocket errors
                                    
                                except Exception as e:
                                    print(f"Error in websocket_sink: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            # Add the custom sink to Loguru logger
                            from app.logger import logger
                            sink_id = logger.add(
                                websocket_sink, 
                                level="INFO", 
                                format="{message}",
                                enqueue=True  # Enable thread-safe enqueueing
                            )
                            
                            # Send startup confirmation message
                            startup_msg = {
                                "type": "progress",
                                "status": "processing",
                                "message": "🤖 [INFO] Agent thoughts streaming initialized - starting execution",
                                "progress": 45
                            }
                            await websocket.send_text(json.dumps(startup_msg))
                            print("DEBUG: Sent startup confirmation message")
                            
                            # Initialize agent thoughts streaming
                            logger.info("🚀 Manus agent starting - thoughts will stream to UI")
                            
                            # Execute the agent
                            try:
                                # Send a message indicating agent execution is starting
                                logger.info(f"🎯 Starting to process: {chat_request.message[:100]}...")
                                
                                # Run the agent
                                await agent.run(chat_request.message)
                                
                                # Send completion message
                                logger.info("✅ Agent execution completed successfully")
                                
                            except Exception as agent_error:
                                logger.error(f"❌ Agent execution failed: {str(agent_error)}")
                                raise
                            finally:
                                # Always remove the custom Loguru sink
                                try:
                                    logger.remove(sink_id)
                                except Exception:
                                    pass  # Silently handle sink removal errors
                            
                            update_progress(conv_id, "processing", "Finalizing results...", 90)
                            
                            # Send progress update via WebSocket
                            progress_msg = {
                                "type": "progress",
                                "status": "processing",
                                "message": "Finalizing results...",
                                "progress": 90
                            }
                            await websocket.send_text(json.dumps(progress_msg))
                            print(f"DEBUG: Captured {len(captured_logs)} agent thoughts during execution")
                            
                            # Send final summary of captured thoughts
                            if captured_logs:
                                summary_msg = {
                                    "type": "progress",
                                    "status": "processing",
                                    "message": f"🤖 [INFO] Captured {len(captured_logs)} agent thoughts during execution",
                                    "progress": 85
                                }
                                await websocket.send_text(json.dumps(summary_msg))
                                print(f"DEBUG: Sent summary of {len(captured_logs)} captured thoughts")
                            
                            # Format the response
                            if agent_output.strip():
                                response_content = f"🤖 **Manus Agent Results**\n\n{agent_output.strip()}"
                            elif agent_errors.strip():
                                response_content = f"🤖 **Manus Agent Output**\n\n{agent_errors.strip()}"
                            else:
                                response_content = f"🤖 **Manus Agent**\n\nTask completed successfully. The agent processed your request: '{chat_request.message}'"
                            
                            # Clean up agent resources
                            await agent.cleanup()
                                
                        except ImportError as e:
                            response_content = f"🤖 Manus agent not available: {str(e)}. Please ensure all dependencies are installed."
                            update_progress(conv_id, "error", "Manus agent unavailable", 100)
                        except Exception as e:
                            response_content = f"🤖 Manus agent error: {str(e)}. Please try again or contact support."
                            update_progress(conv_id, "error", "Manus agent failed", 100)
                        
                        update_progress(conv_id, "completed", "Manus agent completed", 100)
                        
                    elif task_type == 'web_browsing':
                        # Update progress
                        update_progress(conv_id, "processing", "Initializing web browsing agent...", 10)
                        
                        # Send progress update via WebSocket with debug logging
                        progress_msg = {
                            "type": "progress",
                            "status": "processing",
                            "message": "Initializing web browsing agent...",
                            "progress": 10
                        }
                        await websocket.send_text(json.dumps(progress_msg))
                        print(f"DEBUG: Sent progress update via WebSocket: {progress_msg}")
                        
                        try:
                            # Import browser tool directly (Playwright-based)
                            from app.tool.browser_use_tool import BrowserUseTool
                            import asyncio
                            import re
                            
                            update_progress(conv_id, "processing", "Setting up browser...", 30)
                            
                            # Send progress update via WebSocket with debug logging
                            progress_msg = {
                                "type": "progress",
                                "status": "processing",
                                "message": "Setting up browser...",
                                "progress": 30
                            }
                            await websocket.send_text(json.dumps(progress_msg))
                            print(f"DEBUG: Sent progress update via WebSocket: {progress_msg}")
                            
                            # Create browser tool instance
                            browser_tool = BrowserUseTool()
                            
                            update_progress(conv_id, "processing", "Navigating to website...", 50)
                            
                            # Send progress update via WebSocket with debug logging
                            progress_msg = {
                                "type": "progress",
                                "status": "processing",
                                "message": "Navigating to website...",
                                "progress": 50
                            }
                            await websocket.send_text(json.dumps(progress_msg))
                            print(f"DEBUG: Sent progress update via WebSocket: {progress_msg}")
                            
                            # Extract URL from user message if present
                            url_match = re.search(r'https?://[^\s]+', chat_request.message)
                            if url_match:
                                url = url_match.group(0)
                                
                                # Navigate to the URL
                                nav_result = await browser_tool.execute(
                                    action="go_to_url",
                                    url=url
                                )
                                
                                update_progress(conv_id, "processing", "Extracting content...", 70)
                                
                                # Send progress update via WebSocket
                                progress_msg = {
                                    "type": "progress",
                                    "status": "processing",
                                    "message": "Extracting content...",
                                    "progress": 70
                                }
                                await websocket.send_text(json.dumps(progress_msg))
                                print(f"DEBUG: Sent progress update via WebSocket: {progress_msg}")
                                
                                # Extract content based on user request
                                extract_result = await browser_tool.execute(
                                    action="extract_content",
                                    goal=chat_request.message
                                )
                                
                                response_content = f"🌐 **Web Browsing Results for {url}**\n\n{extract_result.output}"
                                
                            else:
                                # If no URL found, perform web search
                                search_result = await browser_tool.execute(
                                    action="web_search",
                                    query=chat_request.message
                                )
                                
                                update_progress(conv_id, "processing", "Analyzing search results...", 70)
                                
                                # Send progress update via WebSocket
                                progress_msg = {
                                    "type": "progress",
                                    "status": "processing",
                                    "message": "Analyzing search results...",
                                    "progress": 70
                                }
                                await websocket.send_text(json.dumps(progress_msg))
                                print(f"DEBUG: Sent progress update via WebSocket: {progress_msg}")
                                
                                # Extract content from search results
                                extract_result = await browser_tool.execute(
                                    action="extract_content",
                                    goal=f"Find information about: {chat_request.message}"
                                )
                                
                                response_content = f"🌐 **Web Search Results for '{chat_request.message}'**\n\n{extract_result.output}"
                            
                            update_progress(conv_id, "completed", "Web browsing completed", 100)
                            
                        except asyncio.TimeoutError:
                            response_content = "🌐 Web browsing task timed out after 5 minutes. Please try a simpler request or break it into smaller parts."
                            update_progress(conv_id, "completed", "Web browsing timed out", 100)
                        except ImportError as e:
                            response_content = f"🌐 Web browsing agent not available: {str(e)}. Please ensure all dependencies are installed."
                            update_progress(conv_id, "error", "Web browsing agent unavailable", 100)
                        except Exception as e:
                            response_content = f"🌐 Web browsing error: {str(e)}. Please try again or contact support."
                            update_progress(conv_id, "error", "Web browsing failed", 100)
                        
                    else:
                        # Handle other task types with basic LLM
                        # Always create new LLM instance to support model switching
                        config_name = "default"
                        if chat_request.model and chat_request.model != "auto":
                            # Handle different model formats
                            if chat_request.model.startswith("openrouter/"):
                                config_name = "openrouter"
                            elif "/" in chat_request.model:
                                # Format: provider/model -> provider_model
                                config_name = chat_request.model.replace("/", "_").replace("-", "_")
                            else:
                                config_name = chat_request.model
                        
                        print(f"DEBUG: WebSocket using LLM config: {config_name} for model: {chat_request.model}")
                        current_llm = LLM(config_name=config_name)
                        
                        system_prompts = {
                            'general_chat': "You are a helpful AI assistant. Provide clear, concise, and helpful responses to user questions and requests. Engage in natural conversation and be friendly and informative.",
                            'code_generation': "You are an expert software developer and coding assistant. Help users write, debug, and optimize code. Provide clear explanations, follow best practices, and include comments in your code examples. Focus on writing clean, efficient, and maintainable code.",
                            'reasoning': "You are a logical reasoning expert. Help users think through complex problems step-by-step. Break down problems into smaller parts, analyze relationships, and provide clear, logical explanations for your conclusions. Focus on critical thinking and problem-solving.",
                            'creative_writing': "You are a creative writing assistant. Help users with storytelling, creative content, poetry, and imaginative writing. Be creative, engaging, and help develop compelling narratives, characters, and ideas. Focus on creativity and literary quality."
                        }
                        
                        system_message = Message.system_message(
                            system_prompts.get(task_type, system_prompts['general_chat'])
                        )
                        
                        response_content = await current_llm.ask(
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
                        "type": "message",
                        "content": response_content,
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
