#!/usr/bin/env python3
"""
Minimal browser agent: Gemini Flash + Playwright headless browser.

    pip install google-genai playwright && playwright install chromium
"""

import os, json, asyncio
from playwright.async_api import async_playwright, Page
from google import genai
from google.genai import types

GOOGLE_API_KEY = "USE_YOUR_API_KEY"
MODEL = "gemini-3-flash-preview"
MAX_STEPS = 50
OUTPUT_DIR = "./browser_output"
TASK = "Open Yahoo! Search, search for 'Python web scraping tutorial' and save the top three search results as PDF files."

SELECTOR = 'a, button, input, textarea, select, [role="button"], [onclick]'

TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(name=n, description=d, parameters_json_schema=s)
    for n, d, s in [
        ("browser_navigate", "Navigate to a URL", {
            "type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}),
        ("browser_snapshot", "Get page state with interactive element indices", {
            "type": "object", "properties": {}}),
        ("browser_click", "Click element by index", {
            "type": "object", "properties": {"index": {"type": "integer"}}, "required": ["index"]}),
        ("browser_type", "Type into an input field", {
            "type": "object", "properties": {
                "index": {"type": "integer"}, "text": {"type": "string"},
                "submit": {"type": "boolean", "description": "Press Enter after typing"}
            }, "required": ["index", "text"]}),
        ("browser_pdf", "Save current page as PDF", {
            "type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}),
        ("browser_back", "Go back", {"type": "object", "properties": {}}),
        ("browser_wait", "Wait N seconds", {
            "type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}),
        ("task_complete", "Mark task done", {
            "type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}),
        ("task_failed", "Mark task failed", {
            "type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}),
    ]
])]


class Browser:
    def __init__(self):
        self.pw = None
        self.browser = None
        self.page: Page = None
        self.elements = []

    async def start(self):
        self.pw = await async_playwright().start()
        self.browser = await self.pw.chromium.launch(headless=True)
        self.page = await self.browser.new_page()

    async def stop(self):
        await self.browser.close()
        await self.pw.stop()

    async def _get_visible_element(self, index: int):
        all_els = await self.page.query_selector_all(SELECTOR)
        visible_idx = 0
        for el in all_els:
            box = await el.bounding_box()
            if box and box["width"] > 0 and box["height"] > 0:
                if visible_idx == index:
                    return el
                visible_idx += 1
        return None

    async def navigate(self, url):
        await self.page.goto(url, wait_until="networkidle")
        return f"Navigated to {url}"

    async def snapshot(self):
        title = await self.page.title()
        self.elements = await self.page.evaluate('''() => {
            const out = [];
            document.querySelectorAll('a, button, input, textarea, select, [role="button"], [onclick]').forEach(el => {
                const r = el.getBoundingClientRect();
                if (!r.width || !r.height || r.top > innerHeight * 2) return;
                out.push({
                    index: out.length,
                    tag: el.tagName.toLowerCase(),
                    text: (el.innerText || el.value || el.placeholder || el.ariaLabel || '').slice(0, 80),
                });
            });
            return out;
        }''')
        text = await self.page.evaluate('() => document.body.innerText.slice(0, 3000)')
        elems = "\n".join(f"[{e['index']}] <{e['tag']}> {e['text'][:50]}" for e in self.elements[:50])
        return f"URL: {self.page.url}\nTitle: {title}\n\nElements:\n{elems}\n\nText:\n{text[:1500]}"

    async def click(self, index):
        el = await self._get_visible_element(index)
        if not el:
            return f"Element {index} not found"
        await el.click()
        await self.page.wait_for_load_state("networkidle")
        return f"Clicked [{index}]"

    async def type_text(self, index, text, submit=False):
        el = await self._get_visible_element(index)
        if not el:
            return f"Element {index} not found"
        await el.fill(text)
        if submit:
            await el.press("Enter")
            await self.page.wait_for_load_state("networkidle")
        return f"Typed '{text}'"

    async def save_pdf(self, filename):
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        path = os.path.join(OUTPUT_DIR, filename)
        await self.page.pdf(path=path)
        return f"Saved {path}"


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = genai.Client(api_key=GOOGLE_API_KEY)
    browser = Browser()
    await browser.start()

    history = [types.Content(role="user", parts=[types.Part.from_text(text=f"""You are a browser automation agent. Task: {TASK}

1. Navigate to a site, then snapshot to see elements.
2. Use element indices to click/type. Snapshot after actions.
3. Save PDFs to {OUTPUT_DIR}. Call task_complete or task_failed when done.""")])]

    done = False
    for step in range(1, MAX_STEPS + 1):
        if done:
            break
        print(f"\n--- Step {step} ---")

        resp = client.models.generate_content(
            model=MODEL, contents=history,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY"))))

        if not resp.candidates:
            break
        history.append(resp.candidates[0].content)

        if not resp.function_calls:
            history.append(types.Content(role="user", parts=[types.Part.from_text(text="Continue.")]))
            continue

        parts = []
        for fc in resp.function_calls:
            name, args = fc.name, dict(fc.args or {})
            print(f"  {name}({json.dumps(args)[:120]})")

            if name == "task_complete":
                print(f"  DONE: {args.get('summary')}")
                done = True
                result = "OK"
            elif name == "task_failed":
                print(f"  FAILED: {args.get('reason')}")
                done = True
                result = "OK"
            elif name == "browser_navigate":
                result = await browser.navigate(args["url"])
            elif name == "browser_snapshot":
                result = await browser.snapshot()
            elif name == "browser_click":
                result = await browser.click(args["index"])
            elif name == "browser_type":
                result = await browser.type_text(args["index"], args["text"], args.get("submit", False))
            elif name == "browser_pdf":
                result = await browser.save_pdf(args["filename"])
            elif name == "browser_back":
                await browser.page.go_back()
                await browser.page.wait_for_load_state("networkidle")
                result = "Went back"
            elif name == "browser_wait":
                await asyncio.sleep(min(max(args.get("seconds", 1), 1), 10))
                result = "Waited"
            else:
                result = f"Unknown: {name}"

            print(f"  -> {result[:200]}")
            parts.append(types.Part.from_function_response(name=name, response={"result": result}))
            if done:
                break

        history.append(types.Content(role="user", parts=parts))

    await browser.stop()


if __name__ == "__main__":
    asyncio.run(main())
