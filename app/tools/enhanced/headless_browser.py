"""
Headless Browser Automation for OpenManus
Comprehensive web automation, scraping, and interaction capabilities
"""

import asyncio
import json
import logging
import base64
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import re

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = BrowserContext = Page = ElementHandle = None


class BrowserType(Enum):
    """Supported browser types"""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class WaitCondition(Enum):
    """Wait conditions for page operations"""
    LOAD = "load"
    DOMCONTENTLOADED = "domcontentloaded"
    NETWORKIDLE = "networkidle"
    COMMIT = "commit"


class ScreenshotFormat(Enum):
    """Screenshot formats"""
    PNG = "png"
    JPEG = "jpeg"


@dataclass
class BrowserConfig:
    """Browser configuration"""
    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: Optional[str] = None
    timeout: int = 30000  # milliseconds
    navigation_timeout: int = 30000
    slow_mo: int = 0  # milliseconds delay between actions
    downloads_path: Optional[str] = None
    proxy: Optional[Dict[str, str]] = None
    ignore_https_errors: bool = True
    extra_http_headers: Dict[str, str] = field(default_factory=dict)
    locale: str = "en-US"
    timezone: str = "UTC"
    geolocation: Optional[Dict[str, float]] = None
    permissions: List[str] = field(default_factory=list)


@dataclass
class PageInfo:
    """Page information"""
    url: str
    title: str
    content: str
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: List[Dict[str, Any]] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ElementInfo:
    """Element information"""
    tag_name: str
    text: str
    attributes: Dict[str, str] = field(default_factory=dict)
    bounding_box: Optional[Dict[str, float]] = None
    is_visible: bool = False
    is_enabled: bool = False


@dataclass
class FormData:
    """Form data for filling"""
    selector: str
    value: str
    input_type: str = "text"  # text, select, checkbox, radio, file
    wait_for_element: bool = True
    clear_first: bool = True


@dataclass
class NavigationResult:
    """Navigation result"""
    success: bool
    url: str
    status_code: int
    error: Optional[str] = None
    load_time: float = 0.0
    page_info: Optional[PageInfo] = None


class HeadlessBrowser:
    """Headless browser automation system"""
    
    def __init__(self, config: BrowserConfig = None):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is required for browser automation. Install with: pip install playwright")
        
        self.config = config or BrowserConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Browser instances
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Session management
        self.session_cookies: List[Dict[str, Any]] = []
        self.session_storage: Dict[str, str] = {}
        self.local_storage: Dict[str, str] = {}
        
        # Request/response tracking
        self.request_log: List[Dict[str, Any]] = []
        self.response_log: List[Dict[str, Any]] = []
        
        # Screenshots directory
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def start(self) -> None:
        """Start browser instance"""
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser
            browser_launcher = getattr(self.playwright, self.config.browser_type.value)
            
            launch_options = {
                "headless": self.config.headless,
                "slow_mo": self.config.slow_mo,
                "args": [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor"
                ]
            }
            
            if self.config.proxy:
                launch_options["proxy"] = self.config.proxy
            
            self.browser = await browser_launcher.launch(**launch_options)
            
            # Create context
            context_options = {
                "viewport": {
                    "width": self.config.viewport_width,
                    "height": self.config.viewport_height
                },
                "ignore_https_errors": self.config.ignore_https_errors,
                "locale": self.config.locale,
                "timezone_id": self.config.timezone,
                "extra_http_headers": self.config.extra_http_headers
            }
            
            if self.config.user_agent:
                context_options["user_agent"] = self.config.user_agent
            
            if self.config.downloads_path:
                context_options["accept_downloads"] = True
                context_options["downloads_path"] = self.config.downloads_path
            
            if self.config.geolocation:
                context_options["geolocation"] = self.config.geolocation
            
            if self.config.permissions:
                context_options["permissions"] = self.config.permissions
            
            self.context = await self.browser.new_context(**context_options)
            
            # Set timeouts
            self.context.set_default_timeout(self.config.timeout)
            self.context.set_default_navigation_timeout(self.config.navigation_timeout)
            
            # Create page
            self.page = await self.context.new_page()
            
            # Set up request/response logging
            self.page.on("request", self._log_request)
            self.page.on("response", self._log_response)
            
            self.logger.info(f"Browser started: {self.config.browser_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to start browser: {e}")
            raise
    
    async def close(self) -> None:
        """Close browser instance"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            self.logger.info("Browser closed")
            
        except Exception as e:
            self.logger.error(f"Error closing browser: {e}")
    
    async def navigate(self, url: str, wait_until: WaitCondition = WaitCondition.LOAD) -> NavigationResult:
        """Navigate to URL"""
        try:
            start_time = datetime.now()
            
            response = await self.page.goto(url, wait_until=wait_until.value)
            
            end_time = datetime.now()
            load_time = (end_time - start_time).total_seconds()
            
            # Get page info
            page_info = await self.get_page_info()
            
            return NavigationResult(
                success=True,
                url=self.page.url,
                status_code=response.status if response else 200,
                load_time=load_time,
                page_info=page_info
            )
            
        except Exception as e:
            self.logger.error(f"Navigation failed: {e}")
            return NavigationResult(
                success=False,
                url=url,
                status_code=0,
                error=str(e)
            )
    
    async def get_page_info(self) -> PageInfo:
        """Get current page information"""
        try:
            url = self.page.url
            title = await self.page.title()
            content = await self.page.content()
            
            # Get cookies
            cookies = await self.context.cookies()
            
            return PageInfo(
                url=url,
                title=title,
                content=content,
                status_code=200,  # Assume success if we can get info
                cookies=cookies
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get page info: {e}")
            return PageInfo(url="", title="", content="", status_code=0)
    
    async def take_screenshot(self, path: str = None, 
                            format: ScreenshotFormat = ScreenshotFormat.PNG,
                            full_page: bool = False) -> str:
        """Take screenshot of current page"""
        try:
            if not path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = str(self.screenshots_dir / f"screenshot_{timestamp}.{format.value}")
            
            screenshot_options = {
                "path": path,
                "type": format.value,
                "full_page": full_page
            }
            
            await self.page.screenshot(**screenshot_options)
            
            self.logger.info(f"Screenshot saved: {path}")
            return path
            
        except Exception as e:
            self.logger.error(f"Failed to take screenshot: {e}")
            raise
    
    async def extract_text(self, selector: str = None) -> str:
        """Extract text from page or element"""
        try:
            if selector:
                element = await self.page.query_selector(selector)
                if element:
                    return await element.text_content() or ""
                return ""
            else:
                return await self.page.text_content("body") or ""
                
        except Exception as e:
            self.logger.error(f"Failed to extract text: {e}")
            return ""
    
    async def extract_links(self, selector: str = "a") -> List[Dict[str, str]]:
        """Extract links from page"""
        try:
            links = []
            elements = await self.page.query_selector_all(selector)
            
            for element in elements:
                href = await element.get_attribute("href")
                text = await element.text_content()
                
                if href:
                    # Convert relative URLs to absolute
                    if href.startswith("/"):
                        base_url = f"{self.page.url.split('/')[0]}//{self.page.url.split('/')[2]}"
                        href = base_url + href
                    elif not href.startswith("http"):
                        href = f"{self.page.url.rstrip('/')}/{href}"
                    
                    links.append({
                        "url": href,
                        "text": text or "",
                        "title": await element.get_attribute("title") or ""
                    })
            
            return links
            
        except Exception as e:
            self.logger.error(f"Failed to extract links: {e}")
            return []
    
    async def extract_images(self, selector: str = "img") -> List[Dict[str, str]]:
        """Extract images from page"""
        try:
            images = []
            elements = await self.page.query_selector_all(selector)
            
            for element in elements:
                src = await element.get_attribute("src")
                alt = await element.get_attribute("alt")
                title = await element.get_attribute("title")
                
                if src:
                    # Convert relative URLs to absolute
                    if src.startswith("/"):
                        base_url = f"{self.page.url.split('/')[0]}//{self.page.url.split('/')[2]}"
                        src = base_url + src
                    elif not src.startswith("http") and not src.startswith("data:"):
                        src = f"{self.page.url.rstrip('/')}/{src}"
                    
                    images.append({
                        "src": src,
                        "alt": alt or "",
                        "title": title or ""
                    })
            
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to extract images: {e}")
            return []
    
    async def click_element(self, selector: str, wait_for_element: bool = True) -> bool:
        """Click element by selector"""
        try:
            if wait_for_element:
                await self.page.wait_for_selector(selector, timeout=self.config.timeout)
            
            await self.page.click(selector)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to click element {selector}: {e}")
            return False
    
    async def fill_form(self, form_data: List[FormData]) -> bool:
        """Fill form with data"""
        try:
            for field in form_data:
                if field.wait_for_element:
                    await self.page.wait_for_selector(field.selector, timeout=self.config.timeout)
                
                if field.input_type == "text" or field.input_type == "email" or field.input_type == "password":
                    if field.clear_first:
                        await self.page.fill(field.selector, "")
                    await self.page.fill(field.selector, field.value)
                
                elif field.input_type == "select":
                    await self.page.select_option(field.selector, field.value)
                
                elif field.input_type == "checkbox":
                    if field.value.lower() in ["true", "1", "yes"]:
                        await self.page.check(field.selector)
                    else:
                        await self.page.uncheck(field.selector)
                
                elif field.input_type == "radio":
                    await self.page.check(field.selector)
                
                elif field.input_type == "file":
                    await self.page.set_input_files(field.selector, field.value)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to fill form: {e}")
            return False
    
    async def wait_for_element(self, selector: str, timeout: int = None) -> bool:
        """Wait for element to appear"""
        try:
            timeout = timeout or self.config.timeout
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
            
        except Exception as e:
            self.logger.error(f"Element {selector} not found within timeout: {e}")
            return False
    
    async def wait_for_text(self, text: str, timeout: int = None) -> bool:
        """Wait for text to appear on page"""
        try:
            timeout = timeout or self.config.timeout
            await self.page.wait_for_function(
                f"document.body.textContent.includes('{text}')",
                timeout=timeout
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Text '{text}' not found within timeout: {e}")
            return False
    
    async def scroll_to_element(self, selector: str) -> bool:
        """Scroll to element"""
        try:
            element = await self.page.query_selector(selector)
            if element:
                await element.scroll_into_view_if_needed()
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to scroll to element {selector}: {e}")
            return False
    
    async def scroll_page(self, direction: str = "down", pixels: int = None) -> None:
        """Scroll page"""
        try:
            if pixels is None:
                pixels = self.config.viewport_height // 2
            
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {pixels})")
            elif direction == "up":
                await self.page.evaluate(f"window.scrollBy(0, -{pixels})")
            elif direction == "top":
                await self.page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
        except Exception as e:
            self.logger.error(f"Failed to scroll page: {e}")
    
    async def execute_javascript(self, script: str) -> Any:
        """Execute JavaScript on page"""
        try:
            result = await self.page.evaluate(script)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute JavaScript: {e}")
            return None
    
    async def get_element_info(self, selector: str) -> Optional[ElementInfo]:
        """Get information about an element"""
        try:
            element = await self.page.query_selector(selector)
            if not element:
                return None
            
            tag_name = await element.evaluate("el => el.tagName.toLowerCase()")
            text = await element.text_content() or ""
            is_visible = await element.is_visible()
            is_enabled = await element.is_enabled()
            
            # Get attributes
            attributes = await element.evaluate("""
                el => {
                    const attrs = {};
                    for (let attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return attrs;
                }
            """)
            
            # Get bounding box
            bounding_box = await element.bounding_box()
            
            return ElementInfo(
                tag_name=tag_name,
                text=text,
                attributes=attributes,
                bounding_box=bounding_box,
                is_visible=is_visible,
                is_enabled=is_enabled
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get element info: {e}")
            return None
    
    async def download_file(self, url: str, filename: str = None) -> Optional[str]:
        """Download file from URL"""
        try:
            if not self.config.downloads_path:
                downloads_dir = Path("downloads")
                downloads_dir.mkdir(exist_ok=True)
                download_path = str(downloads_dir)
            else:
                download_path = self.config.downloads_path
            
            if not filename:
                filename = url.split("/")[-1] or "download"
            
            full_path = Path(download_path) / filename
            
            # Start download
            async with self.page.expect_download() as download_info:
                await self.page.goto(url)
            
            download = await download_info.value
            await download.save_as(str(full_path))
            
            self.logger.info(f"File downloaded: {full_path}")
            return str(full_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download file: {e}")
            return None
    
    async def set_cookies(self, cookies: List[Dict[str, Any]]) -> None:
        """Set cookies for the browser context"""
        try:
            await self.context.add_cookies(cookies)
            self.session_cookies.extend(cookies)
            
        except Exception as e:
            self.logger.error(f"Failed to set cookies: {e}")
    
    async def get_cookies(self, url: str = None) -> List[Dict[str, Any]]:
        """Get cookies from the browser context"""
        try:
            if url:
                return await self.context.cookies([url])
            else:
                return await self.context.cookies()
                
        except Exception as e:
            self.logger.error(f"Failed to get cookies: {e}")
            return []
    
    async def clear_cookies(self) -> None:
        """Clear all cookies"""
        try:
            await self.context.clear_cookies()
            self.session_cookies.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to clear cookies: {e}")
    
    async def set_local_storage(self, key: str, value: str) -> None:
        """Set local storage item"""
        try:
            await self.page.evaluate(f"localStorage.setItem('{key}', '{value}')")
            self.local_storage[key] = value
            
        except Exception as e:
            self.logger.error(f"Failed to set local storage: {e}")
    
    async def get_local_storage(self, key: str = None) -> Union[str, Dict[str, str]]:
        """Get local storage item(s)"""
        try:
            if key:
                return await self.page.evaluate(f"localStorage.getItem('{key}')")
            else:
                return await self.page.evaluate("""
                    () => {
                        const storage = {};
                        for (let i = 0; i < localStorage.length; i++) {
                            const key = localStorage.key(i);
                            storage[key] = localStorage.getItem(key);
                        }
                        return storage;
                    }
                """)
                
        except Exception as e:
            self.logger.error(f"Failed to get local storage: {e}")
            return {} if key is None else ""
    
    async def intercept_requests(self, pattern: str, handler: Callable) -> None:
        """Intercept and modify requests"""
        try:
            await self.page.route(pattern, handler)
            
        except Exception as e:
            self.logger.error(f"Failed to set up request interception: {e}")
    
    async def block_resources(self, resource_types: List[str]) -> None:
        """Block specific resource types (images, stylesheets, etc.)"""
        try:
            async def block_handler(route):
                if route.request.resource_type in resource_types:
                    await route.abort()
                else:
                    await route.continue_()
            
            await self.page.route("**/*", block_handler)
            
        except Exception as e:
            self.logger.error(f"Failed to block resources: {e}")
    
    def get_request_log(self) -> List[Dict[str, Any]]:
        """Get request log"""
        return self.request_log.copy()
    
    def get_response_log(self) -> List[Dict[str, Any]]:
        """Get response log"""
        return self.response_log.copy()
    
    def clear_logs(self) -> None:
        """Clear request/response logs"""
        self.request_log.clear()
        self.response_log.clear()
    
    def _log_request(self, request) -> None:
        """Log request"""
        self.request_log.append({
            "url": request.url,
            "method": request.method,
            "headers": dict(request.headers),
            "timestamp": datetime.now().isoformat()
        })
    
    def _log_response(self, response) -> None:
        """Log response"""
        self.response_log.append({
            "url": response.url,
            "status": response.status,
            "headers": dict(response.headers),
            "timestamp": datetime.now().isoformat()
        })


# Factory functions

def create_headless_browser(headless: bool = True, 
                          browser_type: BrowserType = BrowserType.CHROMIUM) -> HeadlessBrowser:
    """Create basic headless browser"""
    config = BrowserConfig(
        browser_type=browser_type,
        headless=headless
    )
    return HeadlessBrowser(config)

def create_scraping_browser() -> HeadlessBrowser:
    """Create browser optimized for web scraping"""
    config = BrowserConfig(
        browser_type=BrowserType.CHROMIUM,
        headless=True,
        viewport_width=1920,
        viewport_height=1080,
        timeout=60000,
        navigation_timeout=60000,
        ignore_https_errors=True,
        extra_http_headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
    )
    return HeadlessBrowser(config)

def create_testing_browser() -> HeadlessBrowser:
    """Create browser for automated testing"""
    config = BrowserConfig(
        browser_type=BrowserType.CHROMIUM,
        headless=False,  # Visible for debugging
        viewport_width=1280,
        viewport_height=720,
        slow_mo=100,  # Slow down for visibility
        timeout=30000
    )
    return HeadlessBrowser(config)

