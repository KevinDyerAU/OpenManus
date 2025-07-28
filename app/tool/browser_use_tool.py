import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.llm import LLM
from app.tool.base import BaseTool, ToolResult
from app.tool.web_search import WebSearch


_BROWSER_DESCRIPTION = """\
A powerful browser automation tool that allows interaction with web pages through various actions.
* This tool provides commands for controlling a browser session, navigating web pages, and extracting information
* It maintains state across calls, keeping the browser session alive until explicitly closed
* Use this when you need to browse websites, fill forms, click buttons, extract content, or perform web searches
* Each action requires specific parameters as defined in the tool's dependencies

Key capabilities include:
* Navigation: Go to specific URLs, go back, search the web, or refresh pages
* Interaction: Click elements, input text, select from dropdowns, send keyboard commands
* Scrolling: Scroll up/down by pixel amount or scroll to specific text
* Content extraction: Extract and analyze content from web pages based on specific goals
* Tab management: Switch between tabs, open new tabs, or close tabs

Note: When using element indices, refer to the numbered elements shown in the current browser state.
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseTool, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "go_to_url",
                    "click_element",
                    "input_text",
                    "scroll_down",
                    "scroll_up",
                    "scroll_to_text",
                    "send_keys",
                    "get_dropdown_options",
                    "select_dropdown_option",
                    "go_back",
                    "web_search",
                    "wait",
                    "extract_content",
                    "switch_tab",
                    "open_tab",
                    "close_tab",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'go_to_url' or 'open_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click_element', 'input_text', 'get_dropdown_options', or 'select_dropdown_option' actions",
            },
            "text": {
                "type": "string",
                "description": "Text for 'input_text', 'scroll_to_text', or 'select_dropdown_option' actions",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll_down' or 'scroll_up' actions",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
            "query": {
                "type": "string",
                "description": "Search query for 'web_search' action",
            },
            "goal": {
                "type": "string",
                "description": "Extraction goal for 'extract_content' action",
            },
            "keys": {
                "type": "string",
                "description": "Keys to send for 'send_keys' action",
            },
            "seconds": {
                "type": "integer",
                "description": "Seconds to wait for 'wait' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "go_to_url": ["url"],
            "click_element": ["index"],
            "input_text": ["index", "text"],
            "switch_tab": ["tab_id"],
            "open_tab": ["url"],
            "scroll_down": ["scroll_amount"],
            "scroll_up": ["scroll_amount"],
            "scroll_to_text": ["text"],
            "send_keys": ["keys"],
            "get_dropdown_options": ["index"],
            "select_dropdown_option": ["index", "text"],
            "go_back": [],
            "web_search": ["query"],
            "wait": ["seconds"],
            "extract_content": ["goal"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)
    web_search_tool: WebSearch = Field(default_factory=WebSearch, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[LLM] = Field(default_factory=LLM)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            # Windows-specific configuration to avoid subprocess issues
            import platform
            is_windows = platform.system() == "Windows"
            
            browser_config_kwargs = {
                "headless": True,  # Force headless for stability
                "disable_security": True,
                "extra_chromium_args": [
                    "--no-sandbox",  # Disable sandbox (essential for Windows)
                    "--disable-dev-shm-usage",  # Overcome limited resource problems
                    "--disable-gpu",  # Disable GPU acceleration
                    "--disable-web-security",  # Disable web security
                    "--disable-extensions",  # Disable extensions
                    "--disable-plugins",  # Disable plugins
                    "--disable-background-timer-throttling",  # Disable throttling
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--disable-ipc-flooding-protection",
                    "--disable-blink-features=AutomationControlled",  # Hide automation
                    "--disable-default-apps",
                    "--disable-sync",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-component-extensions-with-background-pages",
                ] if is_windows else [
                    "--no-sandbox",  # Also apply key fixes on other platforms
                    "--disable-dev-shm-usage",
                ]
            }

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()
            
            # Set longer timeouts for all platforms to ensure Playwright success
            import platform
            context_config.default_timeout = 90000  # 90 seconds - very generous
            context_config.navigation_timeout = 90000  # 90 seconds for navigation
            
            # Additional Windows-specific timeout settings
            if platform.system() == "Windows":
                context_config.expect_timeout = 30000  # 30 seconds for element expectations

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config
                # Still apply Windows timeouts even with custom config
                if platform.system() == "Windows":
                    context_config.default_timeout = 60000
                    context_config.navigation_timeout = 60000

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def _http_fallback_extract(self, url: str, goal: str, max_content_length: int) -> ToolResult:
        """Fallback HTTP-based content extraction when Playwright fails."""
        try:
            import aiohttp
            from bs4 import BeautifulSoup
            import json
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        
                        # Parse HTML content
                        soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Extract relevant content
                        title = soup.find('title')
                        title_text = title.get_text().strip() if title else "No title found"
                        
                        # Extract main content sections
                        content_sections = []
                        
                        # Method 1: Look for headings and their content
                        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                        for heading in headings:
                            heading_text = heading.get_text().strip()
                            if heading_text and len(heading_text) > 2:
                                content_sections.append(f"**{heading_text}**")
                        
                        # Method 2: Extract meaningful text content
                        content_containers = soup.find_all(['main', 'article', 'section', 'div', 'p', 'li', 'td', 'th'])
                        
                        all_text_blocks = []
                        for container in content_containers:
                            text = container.get_text().strip()
                            # Filter out very short text and common navigation/footer text
                            if (text and len(text) > 15 and 
                                not any(skip_word in text.lower() for skip_word in 
                                       ['cookie', 'privacy policy', 'terms of service', 'copyright', 
                                        'all rights reserved', 'menu', 'navigation', 'skip to'])):
                                all_text_blocks.append(text)
                        
                        # Remove duplicates while preserving order
                        seen = set()
                        unique_text_blocks = []
                        for text in all_text_blocks:
                            if text not in seen and len(text) > 20:
                                seen.add(text)
                                unique_text_blocks.append(text)
                        
                        # Add unique text blocks to content sections
                        content_sections.extend(unique_text_blocks[:10])  # Limit to first 10 blocks
                        
                        # Create content for LLM processing
                        if content_sections:
                            content_text = "\n\n".join(content_sections)
                            full_content = f"Page Title: {title_text}\n\nContent:\n{content_text}"
                        else:
                            # Fallback to body text
                            body = soup.find('body')
                            if body:
                                all_text = body.get_text(separator='\n', strip=True)
                                lines = [line.strip() for line in all_text.split('\n') if line.strip()]
                                meaningful_lines = [line for line in lines if len(line) > 20]
                                full_content = f"Page Title: {title_text}\n\nContent:\n" + "\n".join(meaningful_lines[:15])
                            else:
                                full_content = f"Page Title: {title_text}\n\nContent: Could not extract meaningful content."
                        
                        # Smart truncation for HTTP fallback content
                        if len(full_content) > max_content_length:
                            # Smart truncation - try to keep complete sentences/paragraphs
                            truncated_content = full_content[:max_content_length]
                            # Find the last complete sentence or paragraph
                            last_period = truncated_content.rfind('.')
                            last_newline = truncated_content.rfind('\n\n')
                            
                            if last_period > max_content_length * 0.8:  # If we can keep 80% and end on sentence
                                full_content = truncated_content[:last_period + 1]
                            elif last_newline > max_content_length * 0.7:  # If we can keep 70% and end on paragraph
                                full_content = truncated_content[:last_newline]
                            else:
                                full_content = truncated_content + "..."
                        
                        # Use LLM to process the extracted content
                        prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.

Page Title: {title_text}
Extraction goal: {goal}

Page content:
{full_content}
"""
                        messages = [{"role": "system", "content": prompt}]
                        
                        # Define extraction function schema
                        extraction_function = {
                            "type": "function",
                            "function": {
                                "name": "extract_content",
                                "description": "Extract specific information from a webpage based on a goal",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "extracted_content": {
                                            "type": "object",
                                            "description": "The content extracted from the page according to the goal",
                                            "properties": {
                                                "text": {
                                                    "type": "string",
                                                    "description": "Text content extracted from the page",
                                                },
                                                "metadata": {
                                                    "type": "object",
                                                    "description": "Additional metadata about the extracted content",
                                                    "properties": {
                                                        "source": {
                                                            "type": "string",
                                                            "description": "Source of the extracted content",
                                                        }
                                                    },
                                                },
                                            },
                                        }
                                    },
                                    "required": ["extracted_content"],
                                },
                            },
                        }
                        
                        # Use LLM to extract content with required function calling
                        response = await self.llm.ask_tool(
                            messages,
                            tools=[extraction_function],
                            tool_choice="required",
                        )
                        
                        if response and response.tool_calls:
                            args = json.loads(response.tool_calls[0].function.arguments)
                            extracted_content = args.get("extracted_content", {})
                            
                            # Format the extracted content nicely
                            if isinstance(extracted_content, dict):
                                text_content = extracted_content.get('text', '')
                                metadata = extracted_content.get('metadata', {})
                                
                                if text_content:
                                    formatted_output = f"**Page Title:** {title_text}\n\n**Extracted Content (HTTP fallback):**\n{text_content}"
                                    if metadata and metadata.get('source'):
                                        formatted_output += f"\n\n**Source:** {metadata['source']}"
                                    return ToolResult(output=formatted_output)
                            
                            # Fallback formatting
                            return ToolResult(
                                output=f"**Page Title:** {title_text}\n\n**Extracted Content (HTTP fallback):**\n{str(extracted_content)}"
                            )
                        
                        # If LLM extraction fails, return more content (up to 3000 chars)
                        return ToolResult(
                            output=f"**Page Title:** {title_text}\n\n**Raw Content (HTTP fallback):**\n{full_content[:3000]}{'...' if len(full_content) > 3000 else ''}"
                        )
                    
                    else:
                        return ToolResult(
                            error=f"HTTP fallback failed - Status {response.status}",
                            output=f"Both Playwright and HTTP fallback failed to access {url}"
                        )
                        
        except Exception as e:
            return ToolResult(
                error=f"HTTP fallback extraction failed: {str(e)}",
                output=f"Both Playwright and HTTP fallback methods failed for {url}"
            )

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        query: Optional[str] = None,
        goal: Optional[str] = None,
        keys: Optional[str] = None,
        seconds: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action or search query
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            query: Search query for Google search
            goal: Extraction goal for content extraction
            keys: Keys to send for keyboard actions
            seconds: Seconds to wait
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()
                
                # Store URL for potential fallback
                current_url = None
                if action in ["go_to_url", "extract_content"] and url:
                    current_url = url

                # Get max content length from config - increased for better extraction
                max_content_length = getattr(
                    config.browser_config, "max_content_length", 15000  # Increased from 2000 to 15000
                )

                # Navigation actions
                if action == "go_to_url":
                    if not url:
                        return ToolResult(
                            error="URL is required for 'go_to_url' action"
                        )
                    
                    # Aggressive retry logic to ensure Playwright succeeds
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            page = await context.get_current_page()
                            
                            # Progressive timeout increase with each retry
                            timeout = 90000 + (attempt * 30000)  # 90s, 120s, 150s
                            
                            print(f"Playwright navigation attempt {attempt + 1}/{max_retries} to {url} (timeout: {timeout}ms)")
                            
                            # Try navigation with extended timeout
                            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                            
                            # Wait for page to be ready
                            await page.wait_for_load_state("domcontentloaded", timeout=30000)
                            
                            print(f"✅ Playwright navigation successful on attempt {attempt + 1}")
                            return ToolResult(output=f"Navigated to {url} (Playwright - attempt {attempt + 1})")
                            
                        except Exception as e:
                            print(f"❌ Playwright navigation attempt {attempt + 1} failed: {str(e)}")
                            
                            if attempt == max_retries - 1:
                                # Only use HTTP fallback after all Playwright attempts fail
                                print(f"🔄 All Playwright attempts failed, using HTTP fallback for {url}")
                                if current_url:
                                    fallback_result = await self._http_fallback_extract(current_url, f"Navigate to and extract basic content from {url}", 2000)
                                    fallback_result.output = f"Navigation completed via HTTP fallback: {fallback_result.output}"
                                    return fallback_result
                                else:
                                    return ToolResult(
                                        error=f"All Playwright navigation attempts failed: {str(e)}",
                                        output=f"Failed to navigate to {url} after {max_retries} attempts"
                                    )
                            else:
                                # Wait before retry
                                await asyncio.sleep(2)

                elif action == "go_back":
                    await context.go_back()
                    return ToolResult(output="Navigated back")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                elif action == "web_search":
                    if not query:
                        return ToolResult(
                            error="Query is required for 'web_search' action"
                        )
                    # Execute the web search and return results directly without browser navigation
                    search_response = await self.web_search_tool.execute(
                        query=query, fetch_content=True, num_results=1
                    )
                    # Navigate to the first search result
                    first_search_result = search_response.results[0]
                    url_to_navigate = first_search_result.url

                    page = await context.get_current_page()
                    await page.goto(url_to_navigate)
                    await page.wait_for_load_state()

                    return search_response

                # Element interaction actions
                elif action == "click_element":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'click_element' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "scroll_down" or action == "scroll_up":
                    direction = 1 if action == "scroll_down" else -1
                    amount = (
                        scroll_amount
                        if scroll_amount is not None
                        else context.config.browser_window_size["height"]
                    )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {direction * amount});"
                    )
                    return ToolResult(
                        output=f"Scrolled {'down' if direction > 0 else 'up'} by {amount} pixels"
                    )

                elif action == "scroll_to_text":
                    if not text:
                        return ToolResult(
                            error="Text is required for 'scroll_to_text' action"
                        )
                    page = await context.get_current_page()
                    try:
                        locator = page.get_by_text(text, exact=False)
                        await locator.scroll_into_view_if_needed()
                        return ToolResult(output=f"Scrolled to text: '{text}'")
                    except Exception as e:
                        return ToolResult(error=f"Failed to scroll to text: {str(e)}")

                elif action == "send_keys":
                    if not keys:
                        return ToolResult(
                            error="Keys are required for 'send_keys' action"
                        )
                    page = await context.get_current_page()
                    await page.keyboard.press(keys)
                    return ToolResult(output=f"Sent keys: {keys}")

                elif action == "get_dropdown_options":
                    if index is None:
                        return ToolResult(
                            error="Index is required for 'get_dropdown_options' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    options = await page.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                        element.xpath,
                    )
                    return ToolResult(output=f"Dropdown options: {options}")

                elif action == "select_dropdown_option":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'select_dropdown_option' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    page = await context.get_current_page()
                    await page.select_option(element.xpath, label=text)
                    return ToolResult(
                        output=f"Selected option '{text}' from dropdown at index {index}"
                    )

                # Content extraction actions
                elif action == "extract_content":
                    if not goal:
                        return ToolResult(
                            error="Goal is required for 'extract_content' action"
                        )

                    try:
                        page = await context.get_current_page()
                        
                        # Get both HTML and text content for better extraction
                        html_content = await page.content()
                        text_content = await page.inner_text('body')
                        
                        # Get page title for context
                        page_title = await page.title()
                        
                        # Use text content primarily, with HTML as backup
                        if text_content and len(text_content.strip()) > 100:
                            # Use clean text content
                            main_content = text_content.strip()
                        else:
                            # Fallback to markdown conversion of HTML
                            import markdownify
                            main_content = markdownify.markdownify(html_content)
                        
                        # Ensure we don't truncate too aggressively
                        if len(main_content) > max_content_length:
                            # Smart truncation - try to keep complete sentences/paragraphs
                            truncated_content = main_content[:max_content_length]
                            # Find the last complete sentence or paragraph
                            last_period = truncated_content.rfind('.')
                            last_newline = truncated_content.rfind('\n\n')
                            
                            if last_period > max_content_length * 0.8:  # If we can keep 80% and end on sentence
                                main_content = truncated_content[:last_period + 1]
                            elif last_newline > max_content_length * 0.7:  # If we can keep 70% and end on paragraph
                                main_content = truncated_content[:last_newline]
                            else:
                                main_content = truncated_content + "..."
                        
                        prompt = f"""\
Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.

Page Title: {page_title}
Extraction goal: {goal}

Page content:
{main_content}
"""
                        messages = [{"role": "system", "content": prompt}]

                        # Define extraction function schema
                        extraction_function = {
                            "type": "function",
                            "function": {
                                "name": "extract_content",
                                "description": "Extract specific information from a webpage based on a goal",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "extracted_content": {
                                            "type": "object",
                                            "description": "The content extracted from the page according to the goal",
                                            "properties": {
                                                "text": {
                                                    "type": "string",
                                                    "description": "Text content extracted from the page",
                                                },
                                                "metadata": {
                                                    "type": "object",
                                                    "description": "Additional metadata about the extracted content",
                                                    "properties": {
                                                        "source": {
                                                            "type": "string",
                                                            "description": "Source of the extracted content",
                                                        }
                                                    },
                                                },
                                            },
                                        }
                                    },
                                    "required": ["extracted_content"],
                                },
                            },
                        }

                        # Use LLM to extract content with required function calling
                        response = await self.llm.ask_tool(
                            messages,
                            tools=[extraction_function],
                            tool_choice="required",
                        )

                        if response and response.tool_calls:
                            args = json.loads(response.tool_calls[0].function.arguments)
                            extracted_content = args.get("extracted_content", {})
                            
                            # Format the extracted content nicely
                            if isinstance(extracted_content, dict):
                                text_content = extracted_content.get('text', '')
                                metadata = extracted_content.get('metadata', {})
                                
                                if text_content:
                                    formatted_output = f"**Page Title:** {page_title}\n\n**Extracted Content:**\n{text_content}"
                                    if metadata and metadata.get('source'):
                                        formatted_output += f"\n\n**Source:** {metadata['source']}"
                                    return ToolResult(output=formatted_output)
                            
                            # Fallback formatting
                            return ToolResult(
                                output=f"**Page Title:** {page_title}\n\n**Extracted Content:**\n{str(extracted_content)}"
                            )

                        # If LLM extraction fails, return the raw content (first 2000 chars)
                        return ToolResult(
                            output=f"**Page Title:** {page_title}\n\n**Raw Content:**\n{main_content[:2000]}{'...' if len(main_content) > 2000 else ''}"
                        )
                        
                    except Exception as e:
                        # If Playwright extraction fails, try HTTP fallback as last resort
                        print(f"❌ Playwright content extraction failed: {str(e)}")
                        if current_url:
                            print(f"🔄 Using HTTP fallback for content extraction from {current_url}")
                            return await self._http_fallback_extract(current_url, goal, max_content_length)
                        else:
                            return ToolResult(
                                error=f"Content extraction failed: {str(e)}",
                                output="Failed to extract content using Playwright and no URL available for HTTP fallback."
                            )

                # Tab management actions
                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    page = await context.get_current_page()
                    await page.wait_for_load_state()
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "open_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'open_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                # Utility actions
                elif action == "wait":
                    seconds_to_wait = seconds if seconds is not None else 3
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(output=f"Waited for {seconds_to_wait} seconds")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(
        self, context: Optional[BrowserContext] = None
    ) -> ToolResult:
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            # Use provided context or fall back to self.context
            ctx = context or self.context
            if not ctx:
                return ToolResult(error="Browser context not initialized")

            state = await ctx.get_state()

            # Create a viewport_info dictionary if it doesn't exist
            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a screenshot for the state
            page = await ctx.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state()

            screenshot = await page.screenshot(
                full_page=True, animations="disabled", type="jpeg", quality=100
            )

            screenshot = base64.b64encode(screenshot).decode("utf-8")

            # Build the state info with all required fields
            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements": (
                    state.element_tree.clickable_elements_to_string()
                    if state.element_tree
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                    + getattr(state, "pixels_below", 0)
                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return ToolResult(
                output=json.dumps(state_info, indent=4, ensure_ascii=False),
                base64_image=screenshot,
            )
        except Exception as e:
            return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()

    @classmethod
    def create_with_context(cls, context: Context) -> "BrowserUseTool[Context]":
        """Factory method to create a BrowserUseTool with a specific context."""
        tool = cls()
        tool.tool_context = context
        return tool
