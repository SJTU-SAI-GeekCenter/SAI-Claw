"""arXiv paper search tool."""

from __future__ import annotations

import html
import json
import re
import xml.etree.ElementTree as ET
from typing import Any
from urllib.parse import quote_plus

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf"


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


class ArxivSearchTool(Tool):
    """Search arXiv for academic papers."""

    name = "arxiv_search"
    description = (
        "Search arXiv for academic papers. Supports keyword, title, author, "
        "abstract search with category filtering."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (keywords, title, etc.)",
            },
            "category": {
                "type": "string",
                "description": "arXiv category filter (e.g., cs.AI, cs.LG, cs.CL, cs.CV, physics, math)",
            },
            "author": {
                "type": "string",
                "description": "Filter by author name",
            },
            "title": {
                "type": "string",
                "description": "Search in title only",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results (1-20)",
                "minimum": 1,
                "maximum": 20,
                "default": 5,
            },
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                "description": "Sort order",
                "default": "relevance",
            },
        },
        "required": ["query"],
    }

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    async def execute(
        self,
        query: str,
        category: str | None = None,
        author: str | None = None,
        title: str | None = None,
        max_results: int = 5,
        sort_by: str = "relevance",
        **kwargs: Any,
    ) -> str:
        n = min(max(max_results, 1), 20)
        
        # Build search query
        search_parts = []
        if title:
            search_parts.append(f"ti:{title}")
        if author:
            search_parts.append(f"au:{author}")
        if query and not title:
            # Search in title and abstract
            search_parts.append(f"all:{query}")
        
        search_query = " AND ".join(search_parts) if search_parts else f"all:{query}"
        
        # Add category filter
        if category:
            search_query = f"({search_query}) AND cat:{category}"

        # Map sort_by to arXiv API values
        sort_mapping = {
            "relevance": "relevance",
            "lastUpdatedDate": "lastUpdatedDate",
            "submittedDate": "submittedDate",
        }
        sort = sort_mapping.get(sort_by, "relevance")
        sort_order = "descending"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": n,
            "sortBy": sort,
            "sortOrder": sort_order,
        }

        try:
            async with httpx.AsyncClient(proxy=self.proxy, timeout=30.0) as client:
                r = await client.get(ARXIV_API_URL, params=params)
                r.raise_for_status()
            
            papers = self._parse_response(r.text)
            return self._format_results(query, papers, n, category)
            
        except httpx.TimeoutException:
            logger.error("arXiv search timeout")
            return json.dumps({"error": "arXiv API timeout, please try again"}, ensure_ascii=False)
        except Exception as e:
            logger.error("arXiv search failed: {}", e)
            return json.dumps({"error": f"arXiv search failed: {e}"}, ensure_ascii=False)

    def _parse_response(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse arXiv Atom response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            
            # Define namespace
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            
            for entry in root.findall("atom:entry", ns):
                paper = {}
                
                # Title
                title_elem = entry.find("atom:title", ns)
                paper["title"] = _strip_tags(title_elem.text) if title_elem is not None else ""
                
                # Authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)
                paper["authors"] = authors
                
                # Summary/Abstract
                summary = entry.find("atom:summary", ns)
                paper["summary"] = _strip_tags(summary.text) if summary is not None else ""
                
                # Published date
                published = entry.find("atom:published", ns)
                paper["published"] = published.text[:10] if published is not None else ""
                
                # Updated date
                updated = entry.find("atom:updated", ns)
                paper["updated"] = updated.text[:10] if updated is not None else ""
                
                # arXiv ID
                id_elem = entry.find("atom:id", ns)
                if id_elem is not None:
                    arxiv_id = id_elem.text.split("/")[-1]
                    if "v" in arxiv_id:
                        arxiv_id = arxiv_id.split("v")[0]
                    paper["arxiv_id"] = arxiv_id
                    paper["pdf_url"] = f"{ARXIV_PDF_URL}/{arxiv_id}.pdf"
                    paper["arxiv_url"] = f"https://arxiv.org/abs/{arxiv_id}"
                
                # Categories
                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        categories.append(term)
                paper["categories"] = categories
                
                # Primary category
                primary = entry.find("arxiv:primary_category", ns)
                if primary is not None:
                    paper["primary_category"] = primary.get("term", "")
                
                papers.append(paper)
                
        except ET.ParseError as e:
            logger.error("Failed to parse arXiv XML: {}", e)
            
        return papers

    def _format_results(
        self, query: str, papers: list[dict[str, Any]], n: int, category: str | None
    ) -> str:
        """Format papers into readable output."""
        if not papers:
            cat_hint = f" in category '{category}'" if category else ""
            return f"No arXiv papers found for: '{query}'{cat_hint}"
        
        lines = [f"arXiv search results for: '{query}'"]
        if category:
            lines[0] += f" [category: {category}]"
        lines.append(f"Found {len(papers)} paper(s)\n")
        
        for i, p in enumerate(papers[:n], 1):
            title = p.get("title", "Untitled")
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            
            arxiv_id = p.get("arxiv_id", "")
            published = p.get("published", "")
            categories = ", ".join(p.get("categories", [])[:3])
            
            # Truncate abstract
            summary = p.get("summary", "")
            if len(summary) > 300:
                summary = summary[:297] + "..."
            
            lines.append(f"{i}. {title}")
            lines.append(f"   Authors: {authors}")
            lines.append(f"   arXiv: {arxiv_id} | Published: {published}")
            if categories:
                lines.append(f"   Categories: {categories}")
            lines.append(f"   URL: {p.get('arxiv_url', '')}")
            lines.append(f"   PDF: {p.get('pdf_url', '')}")
            if summary:
                lines.append(f"   Abstract: {summary}")
            lines.append("")
        
        return "\n".join(lines)


class ArxivGetTool(Tool):
    """Get paper details by arXiv ID."""

    name = "arxiv_get"
    description = "Get paper details by arXiv ID (e.g., '2305.12345' or 'cs/9901001')"
    parameters = {
        "type": "object",
        "properties": {
            "arxiv_id": {
                "type": "string",
                "description": "arXiv paper ID (e.g., 2305.12345)",
            },
        },
        "required": ["arxiv_id"],
    }

    def __init__(self, proxy: str | None = None):
        self.proxy = proxy

    async def execute(self, arxiv_id: str, **kwargs: Any) -> str:
        # Clean ID
        arxiv_id = arxiv_id.strip()
        if "arxiv.org" in arxiv_id:
            # Extract ID from URL
            arxiv_id = arxiv_id.split("/")[-1].replace(".pdf", "")
        
        params = {
            "id_list": arxiv_id,
            "max_results": 1,
        }

        try:
            async with httpx.AsyncClient(proxy=self.proxy, timeout=30.0) as client:
                r = await client.get(ARXIV_API_URL, params=params)
                r.raise_for_status()
            
            papers = self._parse_response(r.text)
            
            if not papers:
                return json.dumps(
                    {"error": f"Paper not found: {arxiv_id}"}, ensure_ascii=False
                )
            
            paper = papers[0]
            return json.dumps(paper, ensure_ascii=False, indent=2)
            
        except Exception as e:
            logger.error("arXiv get failed: {}", e)
            return json.dumps({"error": f"Failed to get paper: {e}"}, ensure_ascii=False)

    def _parse_response(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse arXiv Atom response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_text)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom",
            }
            
            for entry in root.findall("atom:entry", ns):
                paper = {}
                
                title_elem = entry.find("atom:title", ns)
                paper["title"] = _strip_tags(title_elem.text) if title_elem is not None else ""
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns)
                    if name is not None:
                        authors.append(name.text)
                paper["authors"] = authors
                
                summary = entry.find("atom:summary", ns)
                paper["summary"] = _strip_tags(summary.text) if summary is not None else ""
                
                published = entry.find("atom:published", ns)
                paper["published"] = published.text[:10] if published is not None else ""
                
                id_elem = entry.find("atom:id", ns)
                if id_elem is not None:
                    arxiv_id = id_elem.text.split("/")[-1]
                    if "v" in arxiv_id:
                        arxiv_id = arxiv_id.split("v")[0]
                    paper["arxiv_id"] = arxiv_id
                    paper["pdf_url"] = f"{ARXIV_PDF_URL}/{arxiv_id}.pdf"
                    paper["arxiv_url"] = f"https://arxiv.org/abs/{arxiv_id}"
                
                categories = []
                for cat in entry.findall("atom:category", ns):
                    term = cat.get("term")
                    if term:
                        categories.append(term)
                paper["categories"] = categories
                
                papers.append(paper)
                
        except ET.ParseError as e:
            logger.error("Failed to parse arXiv XML: {}", e)
            
        return papers
