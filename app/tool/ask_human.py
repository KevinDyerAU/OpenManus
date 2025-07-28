import asyncio
from typing import Optional, Callable
from app.tool import BaseTool


class AskHuman(BaseTool):
    """Add a tool to ask human for help via UI or command line."""

    name: str = "ask_human"
    description: str = "Use this tool to ask human for help."
    parameters: str = {
        "type": "object",
        "properties": {
            "inquire": {
                "type": "string",
                "description": "The question you want to ask human.",
            }
        },
        "required": ["inquire"],
    }
    
    # Class variable to store UI callback function
    ui_callback: Optional[Callable[[str], asyncio.Future]] = None
    
    @classmethod
    def set_ui_callback(cls, callback: Callable[[str], asyncio.Future]):
        """Set the UI callback function for asking questions via web interface."""
        cls.ui_callback = callback

    async def execute(self, inquire: str) -> str:
        if self.ui_callback:
            # Use UI callback if available (web interface)
            try:
                response = await self.ui_callback(inquire)
                return response.strip() if response else "No response provided"
            except Exception as e:
                # Fallback to command line if UI callback fails
                print(f"UI callback failed: {e}, falling back to command line")
                return input(f"""Bot: {inquire}\n\nYou: """).strip()
        else:
            # Fallback to command line input
            return input(f"""Bot: {inquire}\n\nYou: """).strip()
