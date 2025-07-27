"""
Browser Automation Tools for MCP Integration
Provides browser automation capabilities as MCP tools
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import base64
from pathlib import Path

from .headless_browser import (
    HeadlessBrowser, BrowserConfig, BrowserType, WaitCondition, 
    ScreenshotFormat, FormData, NavigationResult, PageInfo, ElementInfo,
    create_headless_browser, create_scraping_browser
)


class BrowserMCPTools:
    """Browser automation tools for MCP"""
    
    def __init__(self, browser: HeadlessBrowser = None):
        self.browser = browser
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tool registry
        self.tools = {
            "browser_navigate": self.navigate,
            "browser_get_page_info": self.get_page_info,
            "browser_take_screenshot": self.take_screenshot,
            "browser_extract_text": self.extract_text,
            "browser_extract_links": self.extract_links,
            "browser_extract_images": self.extract_images,
            "browser_click_element": self.click_element,
            "browser_fill_form": self.fill_form,
            "browser_wait_for_element": self.wait_for_element,
            "browser_wait_for_text": self.wait_for_text,
            "browser_scroll_page": self.scroll_page,
            "browser_execute_javascript": self.execute_javascript,
            "browser_get_element_info": self.get_element_info,
            "browser_download_file": self.download_file,
            "browser_set_cookies": self.set_cookies,
            "browser_get_cookies": self.get_cookies,
            "browser_clear_cookies": self.clear_cookies,
            "browser_set_local_storage": self.set_local_storage,
            "browser_get_local_storage": self.get_local_storage,
            "browser_get_request_log": self.get_request_log,
            "browser_get_response_log": self.get_response_log,
            "browser_clear_logs": self.clear_logs,
            "browser_start": self.start_browser,
            "browser_close": self.close_browser,
            "browser_new_page": self.new_page,
            "browser_search_and_extract": self.search_and_extract,
            "browser_monitor_changes": self.monitor_changes,
            "browser_bulk_extract": self.bulk_extract
        }
    
    async def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definitions for browser automation"""
        return [
            {
                "name": "browser_navigate",
                "description": "Navigate to a URL in the browser",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to navigate to"},
                        "wait_until": {"type": "string", "enum": ["load", "domcontentloaded", "networkidle", "commit"], "default": "load"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "browser_get_page_info",
                "description": "Get information about the current page",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "browser_take_screenshot",
                "description": "Take a screenshot of the current page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to save screenshot"},
                        "format": {"type": "string", "enum": ["png", "jpeg"], "default": "png"},
                        "full_page": {"type": "boolean", "default": False}
                    },
                    "required": []
                }
            },
            {
                "name": "browser_extract_text",
                "description": "Extract text from page or specific element",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector for element (optional)"}
                    },
                    "required": []
                }
            },
            {
                "name": "browser_extract_links",
                "description": "Extract all links from the page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "default": "a", "description": "CSS selector for links"}
                    },
                    "required": []
                }
            },
            {
                "name": "browser_extract_images",
                "description": "Extract all images from the page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "default": "img", "description": "CSS selector for images"}
                    },
                    "required": []
                }
            },
            {
                "name": "browser_click_element",
                "description": "Click an element on the page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector for element to click"},
                        "wait_for_element": {"type": "boolean", "default": True}
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "browser_fill_form",
                "description": "Fill a form with data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "form_data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "selector": {"type": "string"},
                                    "value": {"type": "string"},
                                    "input_type": {"type": "string", "default": "text"},
                                    "wait_for_element": {"type": "boolean", "default": True},
                                    "clear_first": {"type": "boolean", "default": True}
                                },
                                "required": ["selector", "value"]
                            }
                        }
                    },
                    "required": ["form_data"]
                }
            },
            {
                "name": "browser_wait_for_element",
                "description": "Wait for an element to appear on the page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector for element"},
                        "timeout": {"type": "integer", "description": "Timeout in milliseconds"}
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "browser_wait_for_text",
                "description": "Wait for specific text to appear on the page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to wait for"},
                        "timeout": {"type": "integer", "description": "Timeout in milliseconds"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "browser_scroll_page",
                "description": "Scroll the page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["up", "down", "top", "bottom"], "default": "down"},
                        "pixels": {"type": "integer", "description": "Number of pixels to scroll"}
                    },
                    "required": []
                }
            },
            {
                "name": "browser_execute_javascript",
                "description": "Execute JavaScript code on the page",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "script": {"type": "string", "description": "JavaScript code to execute"}
                    },
                    "required": ["script"]
                }
            },
            {
                "name": "browser_get_element_info",
                "description": "Get detailed information about an element",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string", "description": "CSS selector for element"}
                    },
                    "required": ["selector"]
                }
            },
            {
                "name": "browser_download_file",
                "description": "Download a file from URL",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL of file to download"},
                        "filename": {"type": "string", "description": "Filename to save as"}
                    },
                    "required": ["url"]
                }
            },
            {
                "name": "browser_search_and_extract",
                "description": "Search for elements and extract data in one operation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to navigate to"},
                        "search_selectors": {
                            "type": "object",
                            "description": "Selectors for different types of content",
                            "properties": {
                                "text": {"type": "string"},
                                "links": {"type": "string"},
                                "images": {"type": "string"},
                                "data": {"type": "string"}
                            }
                        },
                        "take_screenshot": {"type": "boolean", "default": False}
                    },
                    "required": ["url", "search_selectors"]
                }
            },
            {
                "name": "browser_monitor_changes",
                "description": "Monitor page for changes over time",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to monitor"},
                        "selector": {"type": "string", "description": "Element to monitor"},
                        "interval": {"type": "integer", "default": 5, "description": "Check interval in seconds"},
                        "max_checks": {"type": "integer", "default": 10, "description": "Maximum number of checks"}
                    },
                    "required": ["url", "selector"]
                }
            },
            {
                "name": "browser_bulk_extract",
                "description": "Extract data from multiple pages",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "urls": {"type": "array", "items": {"type": "string"}},
                        "extraction_config": {
                            "type": "object",
                            "properties": {
                                "text_selector": {"type": "string"},
                                "links_selector": {"type": "string"},
                                "images_selector": {"type": "string"},
                                "custom_selectors": {"type": "object"}
                            }
                        },
                        "delay_between_pages": {"type": "integer", "default": 1}
                    },
                    "required": ["urls", "extraction_config"]
                }
            }
        ]
    
    async def ensure_browser(self) -> None:
        """Ensure browser is available and started"""
        if not self.browser:
            self.browser = create_scraping_browser()
        
        if not self.browser.page:
            await self.browser.start()
    
    async def navigate(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to URL"""
        try:
            await self.ensure_browser()
            
            url = arguments["url"]
            wait_until = WaitCondition(arguments.get("wait_until", "load"))
            
            result = await self.browser.navigate(url, wait_until)
            
            return {
                "success": result.success,
                "url": result.url,
                "status_code": result.status_code,
                "load_time": result.load_time,
                "error": result.error,
                "page_info": result.page_info.__dict__ if result.page_info else None
            }
            
        except Exception as e:
            self.logger.error(f"Navigation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_page_info(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get page information"""
        try:
            await self.ensure_browser()
            
            page_info = await self.browser.get_page_info()
            
            return {
                "url": page_info.url,
                "title": page_info.title,
                "content_length": len(page_info.content),
                "content_preview": page_info.content[:1000] + "..." if len(page_info.content) > 1000 else page_info.content,
                "status_code": page_info.status_code,
                "cookies_count": len(page_info.cookies),
                "timestamp": page_info.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get page info: {e}")
            return {"error": str(e)}
    
    async def take_screenshot(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Take screenshot"""
        try:
            await self.ensure_browser()
            
            path = arguments.get("path")
            format_str = arguments.get("format", "png")
            full_page = arguments.get("full_page", False)
            
            screenshot_format = ScreenshotFormat(format_str)
            screenshot_path = await self.browser.take_screenshot(path, screenshot_format, full_page)
            
            # Encode screenshot as base64 for MCP response
            with open(screenshot_path, "rb") as f:
                screenshot_data = base64.b64encode(f.read()).decode()
            
            return {
                "success": True,
                "path": screenshot_path,
                "format": format_str,
                "full_page": full_page,
                "data": screenshot_data
            }
            
        except Exception as e:
            self.logger.error(f"Failed to take screenshot: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_text(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from page or element"""
        try:
            await self.ensure_browser()
            
            selector = arguments.get("selector")
            text = await self.browser.extract_text(selector)
            
            return {
                "success": True,
                "text": text,
                "length": len(text),
                "selector": selector
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract text: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_links(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract links from page"""
        try:
            await self.ensure_browser()
            
            selector = arguments.get("selector", "a")
            links = await self.browser.extract_links(selector)
            
            return {
                "success": True,
                "links": links,
                "count": len(links),
                "selector": selector
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract links: {e}")
            return {"success": False, "error": str(e)}
    
    async def extract_images(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract images from page"""
        try:
            await self.ensure_browser()
            
            selector = arguments.get("selector", "img")
            images = await self.browser.extract_images(selector)
            
            return {
                "success": True,
                "images": images,
                "count": len(images),
                "selector": selector
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract images: {e}")
            return {"success": False, "error": str(e)}
    
    async def click_element(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Click element"""
        try:
            await self.ensure_browser()
            
            selector = arguments["selector"]
            wait_for_element = arguments.get("wait_for_element", True)
            
            success = await self.browser.click_element(selector, wait_for_element)
            
            return {
                "success": success,
                "selector": selector,
                "waited": wait_for_element
            }
            
        except Exception as e:
            self.logger.error(f"Failed to click element: {e}")
            return {"success": False, "error": str(e)}
    
    async def fill_form(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Fill form with data"""
        try:
            await self.ensure_browser()
            
            form_data_list = arguments["form_data"]
            form_data = [
                FormData(
                    selector=item["selector"],
                    value=item["value"],
                    input_type=item.get("input_type", "text"),
                    wait_for_element=item.get("wait_for_element", True),
                    clear_first=item.get("clear_first", True)
                )
                for item in form_data_list
            ]
            
            success = await self.browser.fill_form(form_data)
            
            return {
                "success": success,
                "fields_filled": len(form_data)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fill form: {e}")
            return {"success": False, "error": str(e)}
    
    async def wait_for_element(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for element to appear"""
        try:
            await self.ensure_browser()
            
            selector = arguments["selector"]
            timeout = arguments.get("timeout")
            
            success = await self.browser.wait_for_element(selector, timeout)
            
            return {
                "success": success,
                "selector": selector,
                "timeout": timeout
            }
            
        except Exception as e:
            self.logger.error(f"Failed to wait for element: {e}")
            return {"success": False, "error": str(e)}
    
    async def wait_for_text(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for text to appear"""
        try:
            await self.ensure_browser()
            
            text = arguments["text"]
            timeout = arguments.get("timeout")
            
            success = await self.browser.wait_for_text(text, timeout)
            
            return {
                "success": success,
                "text": text,
                "timeout": timeout
            }
            
        except Exception as e:
            self.logger.error(f"Failed to wait for text: {e}")
            return {"success": False, "error": str(e)}
    
    async def scroll_page(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Scroll page"""
        try:
            await self.ensure_browser()
            
            direction = arguments.get("direction", "down")
            pixels = arguments.get("pixels")
            
            await self.browser.scroll_page(direction, pixels)
            
            return {
                "success": True,
                "direction": direction,
                "pixels": pixels
            }
            
        except Exception as e:
            self.logger.error(f"Failed to scroll page: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_javascript(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute JavaScript"""
        try:
            await self.ensure_browser()
            
            script = arguments["script"]
            result = await self.browser.execute_javascript(script)
            
            return {
                "success": True,
                "script": script,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute JavaScript: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_element_info(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get element information"""
        try:
            await self.ensure_browser()
            
            selector = arguments["selector"]
            element_info = await self.browser.get_element_info(selector)
            
            if element_info:
                return {
                    "success": True,
                    "selector": selector,
                    "element_info": {
                        "tag_name": element_info.tag_name,
                        "text": element_info.text,
                        "attributes": element_info.attributes,
                        "bounding_box": element_info.bounding_box,
                        "is_visible": element_info.is_visible,
                        "is_enabled": element_info.is_enabled
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Element not found",
                    "selector": selector
                }
            
        except Exception as e:
            self.logger.error(f"Failed to get element info: {e}")
            return {"success": False, "error": str(e)}
    
    async def download_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Download file"""
        try:
            await self.ensure_browser()
            
            url = arguments["url"]
            filename = arguments.get("filename")
            
            file_path = await self.browser.download_file(url, filename)
            
            return {
                "success": file_path is not None,
                "url": url,
                "file_path": file_path,
                "filename": filename
            }
            
        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            return {"success": False, "error": str(e)}
    
    async def set_cookies(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Set cookies"""
        try:
            await self.ensure_browser()
            
            cookies = arguments["cookies"]
            await self.browser.set_cookies(cookies)
            
            return {
                "success": True,
                "cookies_set": len(cookies)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to set cookies: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_cookies(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get cookies"""
        try:
            await self.ensure_browser()
            
            url = arguments.get("url")
            cookies = await self.browser.get_cookies(url)
            
            return {
                "success": True,
                "cookies": cookies,
                "count": len(cookies),
                "url": url
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cookies: {e}")
            return {"success": False, "error": str(e)}
    
    async def clear_cookies(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Clear cookies"""
        try:
            await self.ensure_browser()
            
            await self.browser.clear_cookies()
            
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Failed to clear cookies: {e}")
            return {"success": False, "error": str(e)}
    
    async def set_local_storage(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Set local storage"""
        try:
            await self.ensure_browser()
            
            key = arguments["key"]
            value = arguments["value"]
            
            await self.browser.set_local_storage(key, value)
            
            return {
                "success": True,
                "key": key,
                "value": value
            }
            
        except Exception as e:
            self.logger.error(f"Failed to set local storage: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_local_storage(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get local storage"""
        try:
            await self.ensure_browser()
            
            key = arguments.get("key")
            storage = await self.browser.get_local_storage(key)
            
            return {
                "success": True,
                "key": key,
                "storage": storage
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get local storage: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_request_log(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get request log"""
        try:
            await self.ensure_browser()
            
            request_log = self.browser.get_request_log()
            
            return {
                "success": True,
                "requests": request_log,
                "count": len(request_log)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get request log: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_response_log(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get response log"""
        try:
            await self.ensure_browser()
            
            response_log = self.browser.get_response_log()
            
            return {
                "success": True,
                "responses": response_log,
                "count": len(response_log)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get response log: {e}")
            return {"success": False, "error": str(e)}
    
    async def clear_logs(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Clear request/response logs"""
        try:
            await self.ensure_browser()
            
            self.browser.clear_logs()
            
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Failed to clear logs: {e}")
            return {"success": False, "error": str(e)}
    
    async def start_browser(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Start browser"""
        try:
            if not self.browser:
                browser_type = BrowserType(arguments.get("browser_type", "chromium"))
                headless = arguments.get("headless", True)
                self.browser = create_headless_browser(headless, browser_type)
            
            await self.browser.start()
            
            return {
                "success": True,
                "browser_type": self.browser.config.browser_type.value,
                "headless": self.browser.config.headless
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start browser: {e}")
            return {"success": False, "error": str(e)}
    
    async def close_browser(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Close browser"""
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
            
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"Failed to close browser: {e}")
            return {"success": False, "error": str(e)}
    
    async def new_page(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create new page/tab"""
        try:
            await self.ensure_browser()
            
            # Create new page in existing context
            new_page = await self.browser.context.new_page()
            
            # Switch to new page
            self.browser.page = new_page
            
            return {
                "success": True,
                "page_count": len(self.browser.context.pages)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create new page: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_and_extract(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Search and extract data in one operation"""
        try:
            await self.ensure_browser()
            
            url = arguments["url"]
            search_selectors = arguments["search_selectors"]
            take_screenshot = arguments.get("take_screenshot", False)
            
            # Navigate to URL
            nav_result = await self.browser.navigate(url)
            if not nav_result.success:
                return {"success": False, "error": f"Navigation failed: {nav_result.error}"}
            
            # Extract data based on selectors
            results = {}
            
            if "text" in search_selectors:
                results["text"] = await self.browser.extract_text(search_selectors["text"])
            
            if "links" in search_selectors:
                results["links"] = await self.browser.extract_links(search_selectors["links"])
            
            if "images" in search_selectors:
                results["images"] = await self.browser.extract_images(search_selectors["images"])
            
            if "data" in search_selectors:
                # Custom data extraction
                elements = await self.browser.page.query_selector_all(search_selectors["data"])
                data_items = []
                for element in elements:
                    text = await element.text_content()
                    attributes = await element.evaluate("""
                        el => {
                            const attrs = {};
                            for (let attr of el.attributes) {
                                attrs[attr.name] = attr.value;
                            }
                            return attrs;
                        }
                    """)
                    data_items.append({"text": text, "attributes": attributes})
                results["data"] = data_items
            
            # Take screenshot if requested
            screenshot_path = None
            if take_screenshot:
                screenshot_path = await self.browser.take_screenshot()
            
            return {
                "success": True,
                "url": url,
                "results": results,
                "screenshot_path": screenshot_path,
                "extraction_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to search and extract: {e}")
            return {"success": False, "error": str(e)}
    
    async def monitor_changes(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor page for changes"""
        try:
            await self.ensure_browser()
            
            url = arguments["url"]
            selector = arguments["selector"]
            interval = arguments.get("interval", 5)
            max_checks = arguments.get("max_checks", 10)
            
            # Navigate to URL
            nav_result = await self.browser.navigate(url)
            if not nav_result.success:
                return {"success": False, "error": f"Navigation failed: {nav_result.error}"}
            
            # Monitor changes
            changes = []
            previous_content = None
            
            for check in range(max_checks):
                try:
                    element = await self.browser.page.query_selector(selector)
                    if element:
                        current_content = await element.text_content()
                        
                        if previous_content is not None and current_content != previous_content:
                            changes.append({
                                "check": check,
                                "timestamp": datetime.now().isoformat(),
                                "previous": previous_content,
                                "current": current_content
                            })
                        
                        previous_content = current_content
                    
                    if check < max_checks - 1:  # Don't sleep after last check
                        await asyncio.sleep(interval)
                        
                except Exception as e:
                    self.logger.warning(f"Error during check {check}: {e}")
            
            return {
                "success": True,
                "url": url,
                "selector": selector,
                "changes": changes,
                "total_checks": max_checks,
                "changes_detected": len(changes)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to monitor changes: {e}")
            return {"success": False, "error": str(e)}
    
    async def bulk_extract(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from multiple pages"""
        try:
            await self.ensure_browser()
            
            urls = arguments["urls"]
            extraction_config = arguments["extraction_config"]
            delay = arguments.get("delay_between_pages", 1)
            
            results = []
            
            for i, url in enumerate(urls):
                try:
                    # Navigate to URL
                    nav_result = await self.browser.navigate(url)
                    if not nav_result.success:
                        results.append({
                            "url": url,
                            "success": False,
                            "error": f"Navigation failed: {nav_result.error}"
                        })
                        continue
                    
                    # Extract data
                    page_data = {"url": url, "success": True}
                    
                    if "text_selector" in extraction_config:
                        page_data["text"] = await self.browser.extract_text(extraction_config["text_selector"])
                    
                    if "links_selector" in extraction_config:
                        page_data["links"] = await self.browser.extract_links(extraction_config["links_selector"])
                    
                    if "images_selector" in extraction_config:
                        page_data["images"] = await self.browser.extract_images(extraction_config["images_selector"])
                    
                    if "custom_selectors" in extraction_config:
                        custom_data = {}
                        for key, selector in extraction_config["custom_selectors"].items():
                            custom_data[key] = await self.browser.extract_text(selector)
                        page_data["custom"] = custom_data
                    
                    results.append(page_data)
                    
                    # Delay between pages
                    if i < len(urls) - 1 and delay > 0:
                        await asyncio.sleep(delay)
                        
                except Exception as e:
                    self.logger.error(f"Error processing URL {url}: {e}")
                    results.append({
                        "url": url,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "success": True,
                "total_urls": len(urls),
                "successful_extractions": len([r for r in results if r.get("success")]),
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Failed to bulk extract: {e}")
            return {"success": False, "error": str(e)}


# Factory function
def create_browser_mcp_tools(browser: HeadlessBrowser = None) -> BrowserMCPTools:
    """Create browser MCP tools"""
    return BrowserMCPTools(browser)

