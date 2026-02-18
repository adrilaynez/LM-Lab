"""
Meta router — catalog and documentation endpoints.
"""

from fastapi import APIRouter

from api.config import API_TITLE, API_VERSION, API_DESCRIPTION

router = APIRouter(tags=["Meta"])


@router.get("/meta")
async def meta():
    """API metadata."""
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs_url": "/docs",
    }
