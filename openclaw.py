#!/usr/bin/env python3
"""
Browser Agent: LLM-controlled headless browser using Playwright Python and Gemini 3 Flash.

Requirements:
    pip install google-genai playwright
    playwright install chromium
"""

import os
import json
import asyncio
from playwright.async_api import async_playwright, Page

from google import genai
from google.genai import types


# ============================================================================
# Configuration
# ============================================================================

GOOGLE_API_KEY = "SET_YOUR_KEY_HERE"
GEMINI_MODEL = "gemini-3-flash-preview"
MAX_STEPS = 50
OUTPUT_DIR = "./browser_output"

# The task for the agent to accomplish
TASK = "Open Google search, search for 'Python web scraping tutorial' and save the top three search results as PDF files."


# ============================================================================
# Tool Definitions for Gemini
# ============================================================================

BROWSER_TOOLS = [
    types.FunctionDeclaration(
        name="browser_navigate",
        description="Navigate to a URL in the browser",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to"}
            },
            "required": ["url"]
        }
    ),
    types.FunctionDeclaration(
        name="browser_snapshot",
        description="Get the current page state. Returns a list of interactive elements with their index numbers. Use the index to interact with elements.",
        parameters_json_schema={"type": "object", "properties": {}}
    ),
    types.FunctionDeclaration(
        name="browser_click",
        description="Click on an element by its index from the snapshot",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Element index from the snapshot"}
            },
            "required": ["index"]
        }
    ),
    types.FunctionDeclaration(
        name="browser_type",
        description="Type text into an input field",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Element index from the snapshot"},
                "text": {"type": "string", "description": "The text to type"},
                "submit": {"type": "boolean", "description": "Whether to press Enter after typing"}
            },
            "required": ["index", "text"]
        }
    ),
    types.FunctionDeclaration(
        name="browser_pdf",
        description="Save the current page as a PDF file",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename for the PDF"}
            },
            "required": ["filename"]
        }
    ),
    types.FunctionDeclaration(
        name="browser_back",
        description="Go back to the previous page",
        parameters_json_schema={"type": "object", "properties": {}}
    ),
    types.FunctionDeclaration(
        name="browser_wait",
        description="Wait for a specified number of seconds",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Number of seconds to wait (1-10)"}
            },
            "required": ["seconds"]
        }
    ),
    types.FunctionDeclaration(
        name="task_complete",
        description="Declare that the task has been completed successfully",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of what was accomplished"}
            },
            "required": ["summary"]
        }
    ),
    types.FunctionDeclaration(
        name="task_failed",
        description="Declare that the task cannot be completed",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Explanation of why the task failed"}
            },
            "required": ["reason"]
        }
    )
]


# ============================================================================
# Browser Controller (Playwright Python)
# ============================================================================

class BrowserController:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.page: Page = None
        self.elements = []  # Cached elements from last snapshot
    
    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
    
    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def navigate(self, url: str) -> str:
        await self.page.goto(url, wait_until="networkidle")
        return f"Navigated to {url}"
    
    async def snapshot(self) -> str:
        """Get page state with interactive elements."""
        url = self.page.url
        title = await self.page.title()
        
        # Extract interactive elements
        self.elements = await self.page.evaluate('''() => {
            const elements = [];
            const selectors = 'a, button, input, textarea, select, [role="button"], [onclick]';
            
            document.querySelectorAll(selectors).forEach((el, idx) => {
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                if (rect.top > window.innerHeight * 2) return;  // Skip elements too far down
                
                elements.push({
                    index: elements.length,
                    tag: el.tagName.toLowerCase(),
                    type: el.type || null,
                    text: (el.innerText || el.value || el.placeholder || el.ariaLabel || '').slice(0, 80),
                    href: el.href || null
                });
            });
            return elements;
        }''')
        
        # Get visible text
        text_content = await self.page.evaluate('() => document.body.innerText.slice(0, 3000)')
        
        # Format output
        elements_str = "\n".join(
            f"[{e['index']}] <{e['tag']}> {e['text'][:50]}"
            for e in self.elements[:50]  # Limit to first 50
        )
        
        return f"""URL: {url}
Title: {title}

Interactive elements:
{elements_str}

Page text (truncated):
{text_content[:1500]}"""
    
    async def click(self, index: int) -> str:
        if index < 0 or index >= len(self.elements):
            return f"Error: Invalid element index {index}"
        
        elem = self.elements[index]
        
        try:
            elements = await self.page.query_selector_all(
                'a, button, input, textarea, select, [role="button"], [onclick]'
            )
            
            visible_idx = 0
            for el in elements:
                box = await el.bounding_box()
                if box and box['width'] > 0 and box['height'] > 0:
                    if visible_idx == index:
                        # Handle potential new tab
                        try:
                            async with self.page.context.expect_page(timeout=3000) as new_page_info:
                                await el.click()
                            new_page = await new_page_info.value
                            self.page = new_page
                            await self.page.wait_for_load_state("networkidle")
                            return f"Clicked element [{index}]: {elem['text'][:30]} (opened new tab)"
                        except:
                            # No new tab, normal click
                            await el.click()
                            await self.page.wait_for_load_state("networkidle")
                            return f"Clicked element [{index}]: {elem['text'][:30]}"
                    visible_idx += 1
            
            return f"Error: Could not find element {index}"
        except Exception as e:
            return f"Error clicking: {str(e)}"
    
    async def type_text(self, index: int, text: str, submit: bool = False) -> str:
        if index < 0 or index >= len(self.elements):
            return f"Error: Invalid element index {index}"
        
        try:
            elements = await self.page.query_selector_all(
                'a, button, input, textarea, select, [role="button"], [onclick]'
            )
            
            visible_idx = 0
            for el in elements:
                box = await el.bounding_box()
                if box and box['width'] > 0 and box['height'] > 0:
                    if visible_idx == index:
                        await el.fill(text)
                        if submit:
                            await el.press("Enter")
                            await self.page.wait_for_load_state("networkidle")
                        return f"Typed '{text}' into element [{index}]"
                    visible_idx += 1
            
            return f"Error: Could not find element {index}"
        except Exception as e:
            return f"Error typing: {str(e)}"
    
    async def save_pdf(self, filename: str) -> str:
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        filepath = os.path.join(OUTPUT_DIR, filename)
        await self.page.pdf(path=filepath)
        return f"Saved PDF to {filepath}"
    
    async def go_back(self) -> str:
        await self.page.go_back()
        await self.page.wait_for_load_state("networkidle")
        return "Went back to previous page"


# ============================================================================
# Browser Agent
# ============================================================================

class BrowserAgent:
    def __init__(self, task: str):
        self.task = task
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.browser = BrowserController()
        self.history: list[types.Content] = []
        self.step = 0
        self.done = False
        self.failed = False
        self.result = ""
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    async def execute_tool(self, name: str, args: dict) -> str:
        if name == "task_complete":
            self.done = True
            self.result = args.get("summary", "Done")
            return f"Task complete: {self.result}"
        
        if name == "task_failed":
            self.failed = True
            self.result = args.get("reason", "Failed")
            return f"Task failed: {self.result}"
        
        if name == "browser_wait":
            seconds = min(max(args.get("seconds", 1), 1), 10)
            await asyncio.sleep(seconds)
            return f"Waited {seconds}s"
        
        if name == "browser_navigate":
            return await self.browser.navigate(args["url"])
        
        if name == "browser_snapshot":
            return await self.browser.snapshot()
        
        if name == "browser_click":
            return await self.browser.click(args["index"])
        
        if name == "browser_type":
            return await self.browser.type_text(
                args["index"], 
                args["text"], 
                args.get("submit", False)
            )
        
        if name == "browser_pdf":
            return await self.browser.save_pdf(args["filename"])
        
        if name == "browser_back":
            return await self.browser.go_back()
        
        return f"Unknown tool: {name}"

    async def run(self):
        print(f"\n{'='*60}")
        print(f"TASK: {self.task}")
        print(f"{'='*60}\n")
        
        await self.browser.start()
        
        try:
            tools = [types.Tool(function_declarations=BROWSER_TOOLS)]
            
            system_prompt = f"""You are a browser automation agent. Your task:

{self.task}

Guidelines:
1. Call browser_navigate first to go to a website
2. Call browser_snapshot to see the page and get element indices
3. Use the index numbers to click or type into elements
4. After actions, call browser_snapshot to see the result
5. Save PDFs with descriptive names like "result_1.pdf"
6. Call task_complete when done, task_failed if stuck

Output directory: {OUTPUT_DIR}
Begin now."""

            self.history.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=system_prompt)]
            ))
            
            while self.step < MAX_STEPS and not self.done and not self.failed:
                self.step += 1
                print(f"\n--- Step {self.step} ---")
                
                try:
                    response = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=self.history,
                        config=types.GenerateContentConfig(
                            tools=tools,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            tool_config=types.ToolConfig(
                                function_calling_config=types.FunctionCallingConfig(mode="ANY")
                            )
                        )
                    )
                    
                    if not response.candidates:
                        print("No response")
                        break
                    
                    self.history.append(response.candidates[0].content)
                    
                    if not response.function_calls:
                        self.history.append(types.Content(
                            role="user",
                            parts=[types.Part.from_text(text="Continue with a tool call.")]
                        ))
                        continue
                    
                    responses = []
                    for fc in response.function_calls:
                        name = fc.name
                        args = dict(fc.args) if fc.args else {}
                        
                        print(f"Tool: {name}")
                        print(f"Args: {json.dumps(args)[:150]}")
                        
                        result = await self.execute_tool(name, args)
                        print(f"Result: {result[:300]}{'...' if len(result) > 300 else ''}")
                        
                        responses.append(types.Part.from_function_response(
                            name=name,
                            response={"result": result}
                        ))
                        
                        if self.done or self.failed:
                            break
                    
                    if responses:
                        self.history.append(types.Content(role="user", parts=responses))
                
                except Exception as e:
                    print(f"Error: {e}")
                    self.history.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"Error: {e}. Try a different approach.")]
                    ))
            
            print(f"\n{'='*60}")
            if self.done:
                print(f"COMPLETED: {self.result}")
            elif self.failed:
                print(f"FAILED: {self.result}")
            else:
                print(f"MAX STEPS REACHED")
            print(f"Steps: {self.step}")
            print(f"{'='*60}\n")
        
        finally:
            await self.browser.stop()


async def main():
    agent = BrowserAgent(TASK)
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
