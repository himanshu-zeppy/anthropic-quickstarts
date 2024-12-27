import os
import re
from typing import Dict

import aiohttp
from anthropic.types.beta import BetaToolUnionParam
from bs4 import BeautifulSoup
from openai import OpenAI

from .base import BaseAnthropicTool, ToolError, ToolResult


class AnalyzerTool(BaseAnthropicTool):
    """Tool for analyzing webpage event tracking implementation."""

    name = "WebpageAnalyzer"
    description = "Analyzes event tracking implementation on a webpage using OpenAI's API. The url must be publicly accessible. The tool will return detailed summary of tracking implementation along with potential gaps and suggested improvements. It should be used for static analysis of webpage source for tracking implementation."

    def __init__(self):
        self.tracking_types = {
            "google_analytics": [],
            "custom_events": [],
            "pixel_tracking": [],
            "event_listeners": [],
        }

    def to_params(self) -> BetaToolUnionParam:
        """Convert tool to Anthropic tool parameters."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the webpage to analyze",
                    },
                },
                "required": ["url"],
            },
            "type": "custom",
            "cache_control": None,
        }

    async def fetch_webpage(self, url: str) -> str:
        """Fetch webpage content safely."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        raise ToolError(
                            f"HTTP error {response.status}: {response.reason}"
                        )
                    return await response.text()
        except aiohttp.ClientError as e:
            raise ToolError(f"Failed to fetch webpage: {str(e)}") from e
        except Exception as e:
            raise ToolError(f"Error fetching webpage: {str(e)}") from e

    def extract_tracking_code(self, html_content: str) -> Dict:
        """Extract tracking-related code from HTML."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Reset tracking data
            self.tracking_types = {k: [] for k in self.tracking_types}

            # Find Google Analytics
            ga_scripts = soup.find_all(
                "script",
                string=re.compile(
                    r"ga\(|gtag|analytics\.js|googletagmanager", re.IGNORECASE
                ),
            )
            self.tracking_types["google_analytics"] = [
                script.string for script in ga_scripts if script.string
            ]

            # Find custom events and listeners
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string:
                    if "addEventListener" in script.string:
                        self.tracking_types["event_listeners"].append(script.string)
                    if any(
                        keyword in script.string.lower()
                        for keyword in ["track", "event", "analytics"]
                    ):
                        self.tracking_types["custom_events"].append(script.string)

            # Find pixel tracking
            pixels = soup.find_all(
                ["img", "iframe"],
                {"src": re.compile(r"pixel|track|beacon|analytics", re.IGNORECASE)},
            )
            self.tracking_types["pixel_tracking"] = [str(pixel) for pixel in pixels]

            return self.tracking_types
        except Exception as e:
            raise ToolError(f"Error extracting tracking code: {str(e)}") from e

    async def analyze_with_openai(self, tracking_data: Dict) -> str:
        """Generate analysis using OpenAI."""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

        prompt = f"""
        Analyze this webpage tracking implementation and provide a detailed summary:

        Google Analytics Implementation:
        {tracking_data['google_analytics']}

        Custom Event Tracking:
        {tracking_data['custom_events']}

        Pixel Tracking:
        {tracking_data['pixel_tracking']}

        Event Listeners:
        {tracking_data['event_listeners']}

        Please provide:
        1. Overview of tracking implementation
        2. List of identified tracking methods
        3. Potential gaps and recommended improvements
        4. Best practices assessment
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in web analytics and tracking implementations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ToolError("OpenAI API returned empty response")
            return content
        except Exception as e:
            raise ToolError(f"OpenAI API error: {str(e)}") from e

    async def __call__(self, **kwargs) -> ToolResult:
        """Execute the webpage analysis tool."""
        try:
            url = kwargs.get("url")

            if not url:
                raise ToolError("URL is required")

            # Fetch and analyze webpage
            html_content = await self.fetch_webpage(url)
            tracking_data = self.extract_tracking_code(html_content)
            analysis = await self.analyze_with_openai(tracking_data)

            # Format the result
            output = f"""
            Webpage Event Tracking Analysis for {url}
            =====================================

            {analysis}

            Tracking Implementation Details:
            - Google Analytics: {len(tracking_data['google_analytics'])} implementations
            - Custom Events: {len(tracking_data['custom_events'])} events
            - Pixel Tracking: {len(tracking_data['pixel_tracking'])} pixels
            - Event Listeners: {len(tracking_data['event_listeners'])} listeners
            """

            return ToolResult(
                output=output.strip(),
                system=f"Successfully analyzed event tracking for {url}",
            )

        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"Unexpected error: {str(e)}")
