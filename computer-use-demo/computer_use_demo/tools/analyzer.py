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

    def detect_google_analytics(self, soup: BeautifulSoup) -> dict:
        """Enhanced Google Analytics detection."""
        ga_patterns = {
            "universal_analytics": [
                r"google-analytics\.com/analytics\.js",
                r"ga\(",
                r"_gaq\.push",
                r"UA-\d+-\d+",
            ],
            "gtag": [
                r"googletagmanager\.com/gtag/js",
                r"gtag\(",
                r"G-[A-Z0-9]+",  # GA4 measurement ID pattern
            ],
            "tag_manager": [r"googletagmanager\.com/gtm\.js", r"GTM-[A-Z0-9]+"],
        }

        results = {
            "implementations": {
                "universal_analytics": [],
                "gtag": [],
                "tag_manager": [],
            },
            "ids": {"ua_ids": [], "ga4_ids": [], "gtm_ids": []},
        }

        # Check script contents and sources
        for analytics_type, patterns in ga_patterns.items():
            # Check script contents
            content_matches = soup.find_all(
                "script", string=re.compile("|".join(patterns), re.IGNORECASE)
            )

            # Check script sources
            source_matches = soup.find_all(
                "script", src=re.compile("|".join(patterns), re.IGNORECASE)
            )

            results["implementations"][analytics_type].extend(
                [
                    script.string if script.string else script.get("src", "")
                    for script in (content_matches + source_matches)
                ]
            )

        # Extract IDs
        all_scripts = str(soup)
        results["ids"]["ua_ids"] = re.findall(r"UA-\d+-\d+", all_scripts)
        results["ids"]["ga4_ids"] = re.findall(r"G-[A-Z0-9]+", all_scripts)
        results["ids"]["gtm_ids"] = re.findall(r"GTM-[A-Z0-9]+", all_scripts)

        return results

    def detect_custom_events(self, soup: BeautifulSoup) -> dict:
        """Enhanced custom events detection."""
        event_patterns = {
            "standard_events": [
                r'addEventListener\([\'"`](click|submit|change)',
                r"onclick=",
                r"onsubmit=",
                r"onchange=",
            ],
            "analytics_events": [
                r"track\(",
                r"trackEvent",
                r"trackPageview",
                r"trackCustom",
            ],
            "data_attributes": [r"data-analytics", r"data-track", r"data-event"],
        }

        results = {
            "event_listeners": [],
            "inline_handlers": [],
            "tracking_calls": [],
            "data_attributes": [],
        }

        # Find script-based events
        all_scripts = soup.find_all("script")
        for script in all_scripts:
            if script.string:
                for pattern in event_patterns["standard_events"]:
                    matches = re.findall(pattern, script.string)
                    if matches:
                        results["event_listeners"].extend(matches)

                for pattern in event_patterns["analytics_events"]:
                    matches = re.findall(pattern, script.string)
                    if matches:
                        results["tracking_calls"].extend(matches)

        # Find inline event handlers
        elements_with_handlers = soup.find_all(
            lambda tag: any(attr for attr in tag.attrs if attr.startswith("on"))
        )
        results["inline_handlers"] = [
            f"{elem.name}: {attr}"
            for elem in elements_with_handlers
            for attr in elem.attrs
            if attr.startswith("on")
        ]

        # Find data attributes for tracking
        elements_with_tracking = soup.find_all(
            lambda tag: any(
                attr
                for attr in tag.attrs
                if any(
                    pattern.replace("data-", "") in attr
                    for pattern in event_patterns["data_attributes"]
                )
            )
        )
        results["data_attributes"] = [
            f"{elem.name}: {attr}"
            for elem in elements_with_tracking
            for attr in elem.attrs
            if any(
                pattern.replace("data-", "") in attr
                for pattern in event_patterns["data_attributes"]
            )
        ]

        return results

    def detect_pixel_tracking(self, soup: BeautifulSoup) -> dict:
        """Enhanced pixel tracking detection."""
        pixel_patterns = {
            "facebook": [r"facebook\.com/tr", r"connect\.facebook\.net", r"fbq\("],
            "linkedin": [r"linkedin\.com/px", r"snap\.licdn\.com"],
            "twitter": [r"static\.ads-twitter\.com", r"platform\.twitter\.com"],
            "generic": [r"pixel", r"beacon", r"track", r"analytics"],
        }

        results = {
            "pixels": {"facebook": [], "linkedin": [], "twitter": [], "generic": []},
            "metadata": {"pixel_ids": []},
        }

        for platform, patterns in pixel_patterns.items():
            # Check image pixels
            img_pixels = soup.find_all(
                "img", src=re.compile("|".join(patterns), re.IGNORECASE)
            )

            # Check iframes
            iframe_pixels = soup.find_all(
                "iframe", src=re.compile("|".join(patterns), re.IGNORECASE)
            )

            # Check script-based pixels
            script_pixels = soup.find_all(
                "script", src=re.compile("|".join(patterns), re.IGNORECASE)
            )

            results["pixels"][platform].extend(
                [{"type": "img", "src": pixel.get("src")} for pixel in img_pixels]
            )
            results["pixels"][platform].extend(
                [{"type": "iframe", "src": pixel.get("src")} for pixel in iframe_pixels]
            )
            results["pixels"][platform].extend(
                [{"type": "script", "src": pixel.get("src")} for pixel in script_pixels]
            )

        # Extract pixel IDs
        all_content = str(soup)
        # Facebook pixel ID
        fb_pixel_ids = re.findall(r'fbq\(\'init\',\s*[\'"](\d+)[\'"]', all_content)
        # LinkedIn Insight Tag
        li_pixel_ids = re.findall(
            r'_linkedin_partner_id\s*=\s*[\'"](\d+)[\'"]', all_content
        )
        # Twitter pixel ID
        tw_pixel_ids = re.findall(r'twq\(\'init\',\s*[\'"](\w+)[\'"]', all_content)

        results["metadata"]["pixel_ids"].extend(
            fb_pixel_ids + li_pixel_ids + tw_pixel_ids
        )

        return results

    def detect_hotjar(self, soup: BeautifulSoup) -> dict:
        """Detect Hotjar implementation patterns."""
        hotjar_patterns = [
            # Hotjar snippet initialization
            r"hotjar|hj\(|hjid|hjsv",
            # Hotjar script source
            r"static\.hotjar\.com",
            # Hotjar configuration
            r"_hjSettings|_hjRuntime|_hjIncludedInSample",
            # Hotjar recording and heatmap
            r"_hjRecordingEnabled|_hjMinimizedPolls",
            # Hotjar feedback and surveys
            r"_hjLocalStorageTest|_hjDonePolls|_hjUserAttributes",
        ]

        # Check script tags
        hj_scripts = soup.find_all(
            "script", string=re.compile("|".join(hotjar_patterns), re.IGNORECASE)
        )

        # Also check script sources
        hj_sources = soup.find_all(
            "script", src=re.compile("|".join(hotjar_patterns), re.IGNORECASE)
        )

        # Combine and deduplicate scripts
        all_scripts = [
            script.string if script.string else script.get("src", "")
            for script in list(set(hj_scripts + hj_sources))
            if script.string or script.get("src")
        ]

        return {
            "scripts": all_scripts,
            "verification": self.verify_hotjar_implementation(soup, all_scripts),
            "version_info": self.get_hotjar_version(all_scripts),
        }

    def get_hotjar_version(self, scripts: list) -> dict:
        """Extract Hotjar version information."""
        version_info: dict[str, str | None] = {"hjid": None, "hjsv": None}

        for script in scripts:
            hjid_match = re.search(r"hjid:(\d+)", str(script))
            hjsv_match = re.search(r"hjsv:(\d+)", str(script))

            if hjid_match:
                version_info["hjid"] = hjid_match.group(1)
            if hjsv_match:
                version_info["hjsv"] = hjsv_match.group(1)

        return version_info

    def verify_hotjar_implementation(self, soup: BeautifulSoup, scripts: list) -> dict:
        """Verify Hotjar implementation completeness."""
        return {
            "has_script": bool(scripts),
            "has_site_id": bool(re.search(r"hjid:\d+", str(scripts))),
            "has_snippet": bool(
                soup.find("script", src=re.compile(r"static\.hotjar\.com"))
            ),
        }

    def extract_tracking_code(self, html_content: str) -> Dict:
        """Enhanced tracking code extraction."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            tracking_results = {
                "google_analytics": self.detect_google_analytics(soup),
                "custom_events": self.detect_custom_events(soup),
                "pixel_tracking": self.detect_pixel_tracking(soup),
                "hotjar_tracking": self.detect_hotjar(soup),
                "metadata": {
                    "total_scripts": len(soup.find_all("script")),
                    "tracking_coverage": {},
                    "implementation_quality": {},
                },
            }

            # Calculate tracking coverage
            tracking_results["metadata"]["tracking_coverage"] = {
                "ga_implemented": bool(
                    tracking_results["google_analytics"]["implementations"][
                        "universal_analytics"
                    ]
                    or tracking_results["google_analytics"]["implementations"]["gtag"]
                ),
                "events_implemented": bool(
                    tracking_results["custom_events"]["event_listeners"]
                    or tracking_results["custom_events"]["tracking_calls"]
                ),
                "pixels_implemented": any(
                    pixels
                    for pixels in tracking_results["pixel_tracking"]["pixels"].values()
                ),
                "hotjar_implemented": bool(
                    tracking_results["hotjar_tracking"]["scripts"]
                ),
            }

            # Basic implementation quality check
            tracking_results["metadata"]["implementation_quality"] = {
                "has_duplicate_ga": len(
                    tracking_results["google_analytics"]["ids"]["ua_ids"]
                )
                > 1,
                "mixing_ga_versions": bool(
                    tracking_results["google_analytics"]["implementations"][
                        "universal_analytics"
                    ]
                    and tracking_results["google_analytics"]["implementations"]["gtag"]
                ),
                "has_inline_handlers": bool(
                    tracking_results["custom_events"]["inline_handlers"]
                ),
                "using_data_attributes": bool(
                    tracking_results["custom_events"]["data_attributes"]
                ),
                "hotjar_properly_configured": tracking_results["hotjar_tracking"][
                    "verification"
                ]["has_script"]
                and tracking_results["hotjar_tracking"]["verification"]["has_site_id"],
            }

            return tracking_results

        except Exception as e:
            raise ToolError(f"Error extracting tracking code: {str(e)}") from e

    def _format_quality_metrics(self, quality_metrics: dict) -> str:
        """Format implementation quality metrics for output."""
        quality_messages = {
            "has_duplicate_ga": "Multiple GA implementations detected (not recommended)",
            "mixing_ga_versions": "Mixed GA versions detected (should standardize)",
            "has_inline_handlers": "Using inline event handlers (consider using addEventListener)",
            "using_data_attributes": "Using data attributes for tracking (good practice)",
            "hotjar_properly_configured": "Hotjar is properly configured with script and site ID",
        }

        return "\n".join(
            f"- {quality_messages[key]}"
            for key, value in quality_metrics.items()
            if value
        )

    async def analyze_with_openai(self, tracking_data: Dict) -> str:
        """Generate analysis using OpenAI."""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

        prompt = f"""
        Analyze this webpage tracking implementation and provide a detailed summary:

        Google Analytics Implementation:
        - Universal Analytics: {tracking_data['google_analytics']['implementations']['universal_analytics']}
        - GA4/GTM: {tracking_data['google_analytics']['implementations']['gtag']}
        - Tag Manager: {tracking_data['google_analytics']['implementations']['tag_manager']}
        - IDs Found: {tracking_data['google_analytics']['ids']}

        Custom Event Tracking:
        - Event Listeners: {tracking_data['custom_events']['event_listeners']}
        - Inline Handlers: {tracking_data['custom_events']['inline_handlers']}
        - Tracking Calls: {tracking_data['custom_events']['tracking_calls']}
        - Data Attributes: {tracking_data['custom_events']['data_attributes']}

        Pixel Tracking:
        {tracking_data['pixel_tracking']['pixels']}
        Pixel IDs: {tracking_data['pixel_tracking']['metadata']['pixel_ids']}

        Hotjar Implementation:
        - Scripts: {tracking_data['hotjar_tracking']['scripts']}
        - Verification: {tracking_data['hotjar_tracking']['verification']}
        - Version Info: {tracking_data['hotjar_tracking']['version_info']}

        Implementation Quality Metrics:
        {tracking_data['metadata']['implementation_quality']}

        Please provide:
        1. Overview of tracking implementation
        2. List of identified tracking methods
        3. Potential gaps and recommended improvements
        4. Best practices assessment
        5. Implementation quality concerns
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in web analytics and tracking implementations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0,
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
            -----------------------------
            Google Analytics:
            - Universal Analytics: {len(tracking_data['google_analytics']['implementations']['universal_analytics'])} implementations
            - Google Tag Manager: {len(tracking_data['google_analytics']['implementations']['tag_manager'])} implementations
            - GA4 (gtag): {len(tracking_data['google_analytics']['implementations']['gtag'])} implementations

            Custom Events:
            - Event Listeners: {len(tracking_data['custom_events']['event_listeners'])} listeners
            - Inline Handlers: {len(tracking_data['custom_events']['inline_handlers'])} handlers
            - Tracking Calls: {len(tracking_data['custom_events']['tracking_calls'])} calls

            Pixel Tracking:
            - Facebook: {len(tracking_data['pixel_tracking']['pixels']['facebook'])} pixels
            - LinkedIn: {len(tracking_data['pixel_tracking']['pixels']['linkedin'])} pixels
            - Twitter: {len(tracking_data['pixel_tracking']['pixels']['twitter'])} pixels
            - Generic: {len(tracking_data['pixel_tracking']['pixels']['generic'])} pixels

            Hotjar Tracking:
            - Scripts Found: {len(tracking_data['hotjar_tracking']['scripts'])}
            - Properly Configured: {tracking_data['hotjar_tracking']['verification']['has_script'] and tracking_data['hotjar_tracking']['verification']['has_site_id']}
            - Site ID Present: {tracking_data['hotjar_tracking']['verification']['has_site_id']}
            - Version Info: {tracking_data['hotjar_tracking']['version_info']}"

            Implementation Quality:
            --------------------
            {self._format_quality_metrics(tracking_data['metadata']['implementation_quality'])}
            """

            return ToolResult(
                output=output.strip(),
                system=f"Successfully analyzed event tracking for {url}",
            )

        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"Unexpected error: {str(e)}")
