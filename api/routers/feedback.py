"""
Feedback Router — collect user feedback per section.

POST /api/v1/feedback         → store feedback JSON + optional screenshot
GET  /api/v1/feedback         → list all feedback (optional ?page= & ?section= filters)
GET  /api/v1/feedback/tree    → structured tree: pages → sections → items
GET  /api/v1/feedback/screenshot/{page}/{section}/{filename}  → serve screenshot image
POST /api/v1/feedback/archive → archive a feedback item (move JSON + screenshots)
POST /api/v1/feedback/unarchive → restore an archived feedback item
POST /api/v1/feedback/pin     → pin/unpin a feedback item
DELETE /api/v1/feedback/{page}/{section}/{basename} → delete feedback item
"""

import base64
import hashlib
import io
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from api.config import (
    GITHUB_TOKEN,
    FEEDBACK_REPO_OWNER,
    FEEDBACK_REPO_NAME,
    FEEDBACK_REPO_BRANCH,
    FEEDBACK_AUTO_COMMIT,
)

router = APIRouter(tags=["feedback"])


def _ai_triage_heuristic(title: str, comment: str, feedback_type: str | None) -> dict:
    text = f"{title}\n{comment}".lower()
    tags: list[str] = []

    def has_any(*words: str) -> bool:
        return any(w in text for w in words)

    # Very lightweight tagging
    if has_any("error", "exception", "traceback", "crash", "fails", "no funciona", "not working"):
        tags.append("bug")
    if has_any("ui", "button", "layout", "css", "responsive", "mobile", "tablet", "pantalla"):
        tags.append("ui")
    if has_any("slow", "lento", "performance", "lag"):
        tags.append("performance")
    if has_any("typo", "spelling", "ortografía"):
        tags.append("typo")
    if feedback_type and feedback_type.lower() in {"bug", "idea", "question"}:
        tags.append(feedback_type.lower())

    # Priority guess
    priority = "medium"
    if has_any("crash", "urgent", "urgente", "block", "bloquea"):
        priority = "high"
    elif has_any("minor", "pequeño", "cosmetic", "detalle"):
        priority = "low"

    # State suggestion
    state = "triaged" if tags else "new"

    summary = (title or "").strip() or (comment or "").strip().split("\n", 1)[0][:120]
    if not summary:
        summary = "Feedback"

    # Deduplicate tags preserving order
    seen = set()
    deduped_tags = []
    for t in tags:
        if t in seen:
            continue
        seen.add(t)
        deduped_tags.append(t)

    return {
        "state": state,
        "tags": deduped_tags,
        "priority": priority,
        "summary": summary,
    }

# ---------------------------------------------------------------------------
#  Storage — organized as data/feedback/{page_id}/{section_id}/{timestamp}.*
# ---------------------------------------------------------------------------

FEEDBACK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVED_FEEDBACK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "feedback_archived"
ARCHIVED_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
BACKUPS_DIR = Path(__file__).resolve().parent.parent.parent / "backups" / "feedback"
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

# ---------------------------------------------------------------------------
#  In-memory rate limit: 1 per IP+section per 30 seconds (or anon_id if available)
# ---------------------------------------------------------------------------

_rate_limit: dict[str, float] = {}
RATE_LIMIT_SECONDS = 30  # 30 seconds

# Valid feedback states
VALID_STATES = {"new", "triaged", "in_progress", "fixed", "wont_fix", "archived"}


def _decode_image_b64(raw_b64: str) -> tuple[bytes, str]:
    """Decode base64 image and infer extension from magic bytes."""
    img_bytes = base64.b64decode(raw_b64)
    if len(img_bytes) < 32:
        raise ValueError("Image payload too small")

    # JPEG
    if img_bytes.startswith(b"\xff\xd8\xff"):
        return img_bytes, ".jpg"
    # PNG
    if img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return img_bytes, ".png"
    # WEBP (RIFF....WEBP)
    if img_bytes.startswith(b"RIFF") and len(img_bytes) >= 12 and img_bytes[8:12] == b"WEBP":
        return img_bytes, ".webp"

    # Default fallback for unknown but decodable image-like payload
    return img_bytes, ".jpg"


def _check_rate_limit(ip: str, page_id: str, section_id: str, anon_id: Optional[str] = None) -> None:
    # Prefer anon_id for more stable rate limiting (survives IP changes)
    if anon_id:
        key_base = hashlib.sha256(anon_id.encode()).hexdigest()[:16]
        key = f"anon:{key_base}:{page_id}:{section_id}"
    else:
        key = f"ip:{ip}:{page_id}:{section_id}"
    
    now = time.time()
    last = _rate_limit.get(key, 0)
    if now - last < RATE_LIMIT_SECONDS:
        remaining = int(RATE_LIMIT_SECONDS - (now - last))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited. Try again in {remaining}s.",
        )
    _rate_limit[key] = now


def _safe_name(s: str) -> str:
    """Sanitize a string for use as a directory/file name."""
    return s.replace("/", "_").replace("\\", "_").replace("..", "_").strip("_") or "unknown"


# ---------------------------------------------------------------------------
#  Schemas
# ---------------------------------------------------------------------------

class FeedbackPayload(BaseModel):
    page_id: str = Field(..., min_length=1, max_length=64)
    section_id: str = Field(..., min_length=1, max_length=64)
    comment: str = Field(..., min_length=1, max_length=2000)
    title: Optional[str] = Field(None, max_length=200)
    anon_id: Optional[str] = Field(None, max_length=64)
    name: Optional[str] = Field(None, max_length=100)
    screenshot_b64: Optional[str] = None          # auto-capture (silent)
    user_screenshot_b64: Optional[str] = None      # user-attached image
    
    # Extended metadata
    feedback_type: Optional[str] = Field(None, max_length=50)  # "bug", "idea", "question"
    url: Optional[str] = Field(None, max_length=500)  # Full URL with path + hash
    user_agent: Optional[str] = Field(None, max_length=500)
    viewport_width: Optional[int] = None
    viewport_height: Optional[int] = None
    theme: Optional[str] = Field(None, max_length=20)  # "dark", "light"
    language: Optional[str] = Field(None, max_length=10)  # "es", "en"
    local_timestamp: Optional[str] = Field(None, max_length=50)  # User's local time


class FeedbackResponse(BaseModel):
    ok: bool
    filename: str


class FeedbackActionPayload(BaseModel):
    page_id: str = Field(..., min_length=1, max_length=64)
    section_id: str = Field(..., min_length=1, max_length=64)
    basename: str = Field(..., min_length=1, max_length=64)
    archived: bool = False


class UpdateStatePayload(FeedbackActionPayload):
    state: str = Field(..., min_length=1, max_length=50)


class UpdateTagsPayload(FeedbackActionPayload):
    tags: list[str] = Field(..., max_length=20)


class UpdateOwnerPayload(FeedbackActionPayload):
    owner: Optional[str] = Field(None, max_length=100)


class PinPayload(FeedbackActionPayload):
    pinned: Optional[bool] = None


class ReadPayload(FeedbackActionPayload):
    read: bool = True


class BulkItemPayload(BaseModel):
    page_id: str = Field(..., min_length=1, max_length=64)
    section_id: str = Field(..., min_length=1, max_length=64)
    basename: str = Field(..., min_length=1, max_length=64)
    archived: bool = False


class BulkActionPayload(BaseModel):
    action: Literal["archive", "unarchive", "delete", "pin", "unpin", "read", "unread"]
    items: list[BulkItemPayload] = Field(..., min_length=1, max_length=500)


class FeedbackActionResponse(BaseModel):
    ok: bool
    message: str


class BulkActionResponse(BaseModel):
    ok: bool
    message: str
    processed: int
    failed: list[dict[str, str]]


class GithubIssuePayload(FeedbackActionPayload):
    repo_owner: str = Field(..., min_length=1, max_length=100)
    repo_name: str = Field(..., min_length=1, max_length=100)
    title: Optional[str] = Field(None, max_length=200)
    body: Optional[str] = None
    labels: list[str] = Field(default_factory=list, max_length=20)
    assignees: list[str] = Field(default_factory=list, max_length=10)
    force_create: bool = False


class GithubIssueResponse(BaseModel):
    ok: bool
    message: str
    created: bool
    title: str
    body: str
    repository: str
    issue_url: Optional[str] = None


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------


def _resolve_feedback_paths(root: Path, page_id: str, section_id: str, basename: str) -> tuple[Path, Path]:
    safe_page = _safe_name(page_id)
    safe_section = _safe_name(section_id)
    safe_basename = _safe_name(basename)
    section_dir = root / safe_page / safe_section
    json_path = section_dir / f"{safe_basename}.json"
    return section_dir, json_path


def _resolve_screenshot_filename(section_dir: Path, basename: str, explicit_name: Optional[str], kind: str) -> Optional[str]:
    # Prefer explicit filename saved in JSON
    if explicit_name:
        p = section_dir / explicit_name
        if p.exists() and p.is_file():
            return explicit_name

    # Legacy fallback: detect any extension for *_auto / *_user
    candidates = sorted(section_dir.glob(f"{basename}_{kind}.*"))
    if candidates:
        return candidates[0].name
    return None


def _read_feedback_json(root: Path, page_id: str, section_id: str, basename: str) -> tuple[Path, dict]:
    _, json_path = _resolve_feedback_paths(root, page_id, section_id, basename)
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Feedback not found")
    try:
        return json_path, json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corrupt feedback JSON: {e}")


def _write_feedback_json(json_path: Path, data: dict) -> None:
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _commit_feedback_to_github(json_path: Path, data: dict, action: str = "create") -> None:
    """Auto-commit feedback to GitHub repo for persistent storage."""
    if not FEEDBACK_AUTO_COMMIT or not GITHUB_TOKEN or not FEEDBACK_REPO_OWNER or not FEEDBACK_REPO_NAME:
        return
    
    try:
        # Build relative path from project root
        project_root = Path(__file__).resolve().parent.parent.parent
        rel_path = json_path.relative_to(project_root).as_posix()
        
        # Read current file content
        content_b64 = base64.b64encode(json_path.read_bytes()).decode("utf-8")
        
        # Build commit message
        page = data.get("page_id", "unknown")
        section = data.get("section_id", "general")
        title = data.get("title") or "(No title)"
        commit_msg = f"[Feedback] {page}/{section}: {title}"
        
        # Get current file SHA if exists (required for updates)
        sha_url = f"https://api.github.com/repos/{FEEDBACK_REPO_OWNER}/{FEEDBACK_REPO_NAME}/contents/{rel_path}?ref={FEEDBACK_REPO_BRANCH}"
        sha_req = urllib.request.Request(sha_url, headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "LM-Lab-Feedback",
        })
        
        file_sha = None
        try:
            with urllib.request.urlopen(sha_req, timeout=8) as resp:
                existing = json.loads(resp.read().decode("utf-8"))
                file_sha = existing.get("sha")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                # Unexpected error, but don't block feedback submission
                return
        
        # Create or update file
        payload = {
            "message": commit_msg,
            "content": content_b64,
            "branch": FEEDBACK_REPO_BRANCH,
        }
        if file_sha:
            payload["sha"] = file_sha
        
        put_url = f"https://api.github.com/repos/{FEEDBACK_REPO_OWNER}/{FEEDBACK_REPO_NAME}/contents/{rel_path}"
        put_req = urllib.request.Request(
            put_url,
            data=json.dumps(payload).encode("utf-8"),
            method="PUT",
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "LM-Lab-Feedback",
            },
        )
        
        with urllib.request.urlopen(put_req, timeout=12) as resp:
            pass  # Success
    except Exception:
        # Don't block feedback submission on GitHub errors
        pass


def _set_feedback_read_state(page_id: str, section_id: str, basename: str, archived: bool, read: bool) -> None:
    root = ARCHIVED_FEEDBACK_DIR if archived else FEEDBACK_DIR
    json_path, data = _read_feedback_json(root, page_id, section_id, basename)
    data["read"] = bool(read)
    data["read_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") if read else None
    _write_feedback_json(json_path, data)


def _set_feedback_pin_state(page_id: str, section_id: str, basename: str, archived: bool, pinned: Optional[bool]) -> bool:
    root = ARCHIVED_FEEDBACK_DIR if archived else FEEDBACK_DIR
    json_path, data = _read_feedback_json(root, page_id, section_id, basename)
    next_pin = (not bool(data.get("pinned", False))) if pinned is None else bool(pinned)
    data["pinned"] = next_pin
    _write_feedback_json(json_path, data)
    return next_pin


def _archive_feedback_item(page_id: str, section_id: str, basename: str, from_archived: bool) -> None:
    src_root = ARCHIVED_FEEDBACK_DIR if from_archived else FEEDBACK_DIR
    dst_root = ARCHIVED_FEEDBACK_DIR
    src_dir, src_json = _resolve_feedback_paths(src_root, page_id, section_id, basename)
    if not src_json.exists():
        raise HTTPException(status_code=404, detail="Feedback not found")

    dst_dir, dst_json = _resolve_feedback_paths(dst_root, page_id, section_id, basename)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst_json.exists():
        raise HTTPException(status_code=409, detail="Archived feedback already exists")

    safe_basename = _safe_name(basename)
    src_json.replace(dst_json)
    for ss in src_dir.glob(f"{safe_basename}_*.*"):
        ss.replace(dst_dir / ss.name)


def _unarchive_feedback_item(page_id: str, section_id: str, basename: str) -> None:
    src_dir, src_json = _resolve_feedback_paths(ARCHIVED_FEEDBACK_DIR, page_id, section_id, basename)
    if not src_json.exists():
        raise HTTPException(status_code=404, detail="Archived feedback not found")

    dst_dir, dst_json = _resolve_feedback_paths(FEEDBACK_DIR, page_id, section_id, basename)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst_json.exists():
        raise HTTPException(status_code=409, detail="Active feedback with same basename already exists")

    safe_basename = _safe_name(basename)
    src_json.replace(dst_json)
    for ss in src_dir.glob(f"{safe_basename}_*.*"):
        ss.replace(dst_dir / ss.name)


def _delete_feedback_item(page: str, section: str, basename: str, archived: bool) -> None:
    root = ARCHIVED_FEEDBACK_DIR if archived else FEEDBACK_DIR
    section_dir, json_path = _resolve_feedback_paths(root, page, section, basename)
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Feedback not found")

    json_path.unlink(missing_ok=True)
    for ss in section_dir.glob(f"{_safe_name(basename)}_*.*"):
        ss.unlink(missing_ok=True)


def _build_tree(root: Path, archived: bool = False) -> dict[str, dict[str, list]]:
    tree: dict[str, dict[str, list]] = {}

    for f in root.rglob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            pg = data.get("page_id", "unknown")
            sec = data.get("section_id", "general")

            if pg not in tree:
                tree[pg] = {}
            if sec not in tree[pg]:
                tree[pg][sec] = []

            rel = f.parent.relative_to(root)
            rel_str = str(rel).replace(chr(92), "/")
            basename = f.stem

            screenshot_name = _resolve_screenshot_filename(
                f.parent,
                basename,
                data.get("screenshot_file"),
                "auto",
            )
            user_screenshot_name = _resolve_screenshot_filename(
                f.parent,
                basename,
                data.get("user_screenshot_file"),
                "user",
            )

            screenshot_url = (
                f"/api/v1/feedback/screenshot/{rel_str}/{screenshot_name}"
                if screenshot_name
                else None
            )
            user_screenshot_url = (
                f"/api/v1/feedback/screenshot/{rel_str}/{user_screenshot_name}"
                if user_screenshot_name
                else None
            )

            tree[pg][sec].append({
                "basename": basename,
                "page_id": pg,
                "section_id": sec,
                "archived": archived,
                "pinned": bool(data.get("pinned", False)),
                "read": bool(data.get("read", False)),
                "read_at": data.get("read_at"),
                "state": data.get("state", "new"),
                "tags": data.get("tags", []),
                "owner": data.get("owner"),
                "timestamp": data.get("timestamp"),
                "title": data.get("title"),
                "comment": data.get("comment"),
                "name": data.get("name"),
                "anon_id": data.get("anon_id"),
                "github_issue_url": data.get("github_issue_url"),
                "github_issue_number": data.get("github_issue_number"),
                "feedback_type": data.get("feedback_type"),
                "url": data.get("url"),
                "viewport_width": data.get("viewport_width"),
                "viewport_height": data.get("viewport_height"),
                "theme": data.get("theme"),
                "language": data.get("language"),
                "local_timestamp": data.get("local_timestamp"),
                "has_screenshot": bool(screenshot_name),
                "screenshot_url": screenshot_url,
                "has_user_screenshot": bool(user_screenshot_name),
                "user_screenshot_url": user_screenshot_url,
            })
        except Exception:
            continue

    for pg in tree:
        for sec in tree[pg]:
            tree[pg][sec].sort(
                key=lambda d: (bool(d.get("pinned", False)), d.get("timestamp", "")),
                reverse=True,
            )
    return tree


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackPayload, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip, payload.page_id, payload.section_id, payload.anon_id)

    now_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_page = _safe_name(payload.page_id)
    safe_section = _safe_name(payload.section_id)

    # Create nested directory: feedback/{page}/{section}/
    section_dir = FEEDBACK_DIR / safe_page / safe_section
    section_dir.mkdir(parents=True, exist_ok=True)

    basename = now_utc

    # Build JSON data (exclude screenshot bytes from stored JSON)
    data = {
        "page_id": payload.page_id,
        "section_id": payload.section_id,
        "comment": payload.comment,
        "title": payload.title,
        "name": payload.name,
        "anon_id": payload.anon_id,
        "timestamp": now_utc,
        "has_screenshot": False,
        "has_user_screenshot": False,
        "pinned": False,
        "read": False,
        "state": "new",
        "tags": [],
        "owner": None,
        "feedback_type": payload.feedback_type,
        "url": payload.url,
        "user_agent": payload.user_agent,
        "viewport_width": payload.viewport_width,
        "viewport_height": payload.viewport_height,
        "theme": payload.theme,
        "language": payload.language,
        "local_timestamp": payload.local_timestamp,
    }

    json_path = section_dir / f"{basename}.json"

    # Save user-attached screenshot
    if payload.user_screenshot_b64:
        try:
            img_bytes, ext = _decode_image_b64(payload.user_screenshot_b64)
            img_path = section_dir / f"{basename}_user{ext}"
            img_path.write_bytes(img_bytes)
            data["user_screenshot_file"] = f"{basename}_user{ext}"
        except Exception:
            data["has_user_screenshot"] = False

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Auto-commit to GitHub for persistence
    _commit_feedback_to_github(json_path, data, action="create")

    return FeedbackResponse(ok=True, filename=f"{safe_page}/{safe_section}/{basename}.json")


@router.get("/feedback")
async def list_feedback(page: Optional[str] = None, section: Optional[str] = None):
    """List all feedback items, optionally filtered by page and section. Newest first."""
    results = []

    # Search all JSON files recursively
    for f in FEEDBACK_DIR.rglob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if page and data.get("page_id") != page:
                continue
            if section and data.get("section_id") != section:
                continue
            # Add relative path info for screenshot access
            rel = f.parent.relative_to(FEEDBACK_DIR)
            data["_path"] = str(rel).replace("\\", "/")
            data["_basename"] = f.stem
            results.append(data)
        except Exception:
            continue

    # Sort newest first
    results.sort(key=lambda d: d.get("timestamp", ""), reverse=True)
    return results


@router.get("/feedback/tree")
async def feedback_tree():
    """
    Return a structured tree: { pages: { [page]: { [section]: [ items ] } } }
    Each item includes comment, name, timestamp, has_screenshot, screenshot_url.
    """
    active = _build_tree(FEEDBACK_DIR, archived=False)
    archived = _build_tree(ARCHIVED_FEEDBACK_DIR, archived=True)
    return {"pages": active, "archived_pages": archived}


@router.post("/feedback/archive", response_model=FeedbackActionResponse)
async def archive_feedback(payload: FeedbackActionPayload):
    _archive_feedback_item(payload.page_id, payload.section_id, payload.basename, payload.archived)
    return FeedbackActionResponse(ok=True, message="Feedback archived")


@router.post("/feedback/unarchive", response_model=FeedbackActionResponse)
async def unarchive_feedback(payload: FeedbackActionPayload):
    _unarchive_feedback_item(payload.page_id, payload.section_id, payload.basename)
    return FeedbackActionResponse(ok=True, message="Feedback restored from archive")


@router.post("/feedback/pin", response_model=FeedbackActionResponse)
async def pin_feedback(payload: PinPayload):
    next_pin = _set_feedback_pin_state(
        payload.page_id,
        payload.section_id,
        payload.basename,
        payload.archived,
        payload.pinned,
    )
    return FeedbackActionResponse(ok=True, message="Feedback pinned" if next_pin else "Feedback unpinned")


@router.post("/feedback/read", response_model=FeedbackActionResponse)
async def mark_feedback_read(payload: ReadPayload):
    _set_feedback_read_state(
        payload.page_id,
        payload.section_id,
        payload.basename,
        payload.archived,
        payload.read,
    )
    return FeedbackActionResponse(ok=True, message="Feedback marked as read" if payload.read else "Feedback marked as unread")


@router.post("/feedback/update-state", response_model=FeedbackActionResponse)
async def update_feedback_state(payload: UpdateStatePayload):
    """Update feedback state (new, triaged, in_progress, fixed, wont_fix)."""
    if payload.state not in VALID_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state. Must be one of: {', '.join(VALID_STATES)}")
    
    root = ARCHIVED_FEEDBACK_DIR if payload.archived else FEEDBACK_DIR
    json_path, data = _read_feedback_json(root, payload.page_id, payload.section_id, payload.basename)
    data["state"] = payload.state
    data["state_updated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write_feedback_json(json_path, data)
    _commit_feedback_to_github(json_path, data, action="update")
    return FeedbackActionResponse(ok=True, message=f"State updated to '{payload.state}'")


@router.post("/feedback/update-tags", response_model=FeedbackActionResponse)
async def update_feedback_tags(payload: UpdateTagsPayload):
    """Update feedback tags/labels."""
    root = ARCHIVED_FEEDBACK_DIR if payload.archived else FEEDBACK_DIR
    json_path, data = _read_feedback_json(root, payload.page_id, payload.section_id, payload.basename)
    data["tags"] = [t.strip() for t in payload.tags if t.strip()][:20]
    data["tags_updated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write_feedback_json(json_path, data)
    _commit_feedback_to_github(json_path, data, action="update")
    return FeedbackActionResponse(ok=True, message="Tags updated")


@router.post("/feedback/update-owner", response_model=FeedbackActionResponse)
async def update_feedback_owner(payload: UpdateOwnerPayload):
    """Assign or unassign feedback owner."""
    root = ARCHIVED_FEEDBACK_DIR if payload.archived else FEEDBACK_DIR
    json_path, data = _read_feedback_json(root, payload.page_id, payload.section_id, payload.basename)
    data["owner"] = payload.owner
    data["owner_updated_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _write_feedback_json(json_path, data)
    _commit_feedback_to_github(json_path, data, action="update")
    return FeedbackActionResponse(ok=True, message=f"Owner set to '{payload.owner}'" if payload.owner else "Owner cleared")


@router.post("/feedback/bulk", response_model=BulkActionResponse)
async def bulk_feedback_action(payload: BulkActionPayload):
    processed = 0
    failed: list[dict[str, str]] = []

    for item in payload.items:
        ident = f"{item.page_id}/{item.section_id}/{item.basename}"
        try:
            if payload.action == "archive":
                _archive_feedback_item(item.page_id, item.section_id, item.basename, item.archived)
            elif payload.action == "unarchive":
                _unarchive_feedback_item(item.page_id, item.section_id, item.basename)
            elif payload.action == "delete":
                _delete_feedback_item(item.page_id, item.section_id, item.basename, item.archived)
            elif payload.action == "pin":
                _set_feedback_pin_state(item.page_id, item.section_id, item.basename, item.archived, True)
            elif payload.action == "unpin":
                _set_feedback_pin_state(item.page_id, item.section_id, item.basename, item.archived, False)
            elif payload.action == "read":
                _set_feedback_read_state(item.page_id, item.section_id, item.basename, item.archived, True)
            elif payload.action == "unread":
                _set_feedback_read_state(item.page_id, item.section_id, item.basename, item.archived, False)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported bulk action: {payload.action}")
            processed += 1
        except HTTPException as e:
            failed.append({"item": ident, "error": str(e.detail)})
        except Exception as e:
            failed.append({"item": ident, "error": str(e)})

    msg = f"Bulk action '{payload.action}' processed {processed}/{len(payload.items)} item(s)"
    return BulkActionResponse(ok=processed > 0, message=msg, processed=processed, failed=failed)


@router.delete("/feedback/{page}/{section}/{basename}", response_model=FeedbackActionResponse)
async def delete_feedback(page: str, section: str, basename: str, archived: bool = False):
    _delete_feedback_item(page, section, basename, archived)
    return FeedbackActionResponse(ok=True, message="Feedback deleted")


@router.post("/feedback/github-issue", response_model=GithubIssueResponse)
async def create_feedback_github_issue(payload: GithubIssuePayload):
    root = ARCHIVED_FEEDBACK_DIR if payload.archived else FEEDBACK_DIR
    json_path, data = _read_feedback_json(root, payload.page_id, payload.section_id, payload.basename)

    existing_url = data.get("github_issue_url")
    title = payload.title or f"[Feedback] {payload.page_id}/{payload.section_id}: {data.get('title') or '(No title)'}"
    body = payload.body or "\n".join([
        "## Feedback report",
        "",
        f"- Page: `{payload.page_id}`",
        f"- Section: `{payload.section_id}`",
        f"- Timestamp (UTC): `{data.get('timestamp') or 'unknown'}`",
        f"- Reporter: `{data.get('name') or 'Anonymous'}`",
        f"- Archived: `{bool(payload.archived)}`",
        "",
        "### Title",
        data.get("title") or "(No title)",
        "",
        "### Comment",
        data.get("comment") or "",
        "",
        "### Metadata",
        f"- Feedback key: `{payload.page_id}/{payload.section_id}/{payload.basename}`",
        f"- Anonymous ID: `{data.get('anon_id') or 'n/a'}`",
        "",
        "### Attachments",
        f"- User screenshot: {'yes' if data.get('user_screenshot_file') else 'no'}",
        f"- Auto screenshot: {'yes' if data.get('screenshot_file') else 'no'}",
    ])

    repo = f"{payload.repo_owner}/{payload.repo_name}"
    if existing_url and not payload.force_create:
        return GithubIssueResponse(
            ok=True,
            message="Feedback is already linked to a GitHub issue. Use force_create to create another.",
            created=False,
            title=title,
            body=body,
            repository=repo,
            issue_url=existing_url,
        )

    if not GITHUB_TOKEN:
        return GithubIssueResponse(
            ok=True,
            message="GITHUB_TOKEN is not configured on the backend. Returning prefilled issue draft only.",
            created=False,
            title=title,
            body=body,
            repository=repo,
            issue_url=existing_url,
        )

    issue_payload = {
        "title": title,
        "body": body,
        "labels": payload.labels,
        "assignees": payload.assignees,
    }
    req = urllib.request.Request(
        f"https://api.github.com/repos/{payload.repo_owner}/{payload.repo_name}/issues",
        data=json.dumps(issue_payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "LM-Lab-Feedback-Viewer",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            created = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"GitHub API error {e.code}: {err_text}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to create GitHub issue: {e}")

    issue_url = created.get("html_url")
    if issue_url:
        data["github_issue_url"] = issue_url
        data["github_issue_number"] = created.get("number")
        data["github_issue_created_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _write_feedback_json(json_path, data)

    return GithubIssueResponse(
        ok=True,
        message="GitHub issue created",
        created=True,
        title=title,
        body=body,
        repository=repo,
        issue_url=issue_url,
    )


@router.get("/feedback/screenshot/{page}/{section}/{filename}")
async def serve_screenshot(page: str, section: str, filename: str):
    """Serve a screenshot image file."""
    safe_page = _safe_name(page)
    safe_section = _safe_name(section)
    # Prevent path traversal
    safe_filename = Path(filename).name
    img_path = FEEDBACK_DIR / safe_page / safe_section / safe_filename
    if not img_path.exists() or not img_path.is_file():
        archived_candidate = ARCHIVED_FEEDBACK_DIR / safe_page / safe_section / safe_filename
        if archived_candidate.exists() and archived_candidate.is_file():
            img_path = archived_candidate
        else:
            raise HTTPException(status_code=404, detail="Screenshot not found")

    suffix = img_path.suffix.lower()
    if suffix == ".jpg" or suffix == ".jpeg":
        media_type = "image/jpeg"
    elif suffix == ".png":
        media_type = "image/png"
    elif suffix == ".webp":
        media_type = "image/webp"
    else:
        media_type = "application/octet-stream"
    return FileResponse(img_path, media_type=media_type)


@router.post("/feedback/sync-from-github")
async def sync_feedbacks_from_github():
    """Sincronizar feedbacks desde GitHub repo al backend local."""
    missing: list[str] = []
    if not GITHUB_TOKEN:
        missing.append("GITHUB_TOKEN")
    if not FEEDBACK_REPO_OWNER:
        missing.append("FEEDBACK_REPO_OWNER")
    if not FEEDBACK_REPO_NAME:
        missing.append("FEEDBACK_REPO_NAME")
    if not FEEDBACK_REPO_BRANCH:
        missing.append("FEEDBACK_REPO_BRANCH")
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"GitHub sync not configured. Missing: {', '.join(missing)}",
        )

    try:
        synced = 0
        errors = []

        # List all files in data/feedback directory from GitHub
        api_url = f"https://api.github.com/repos/{FEEDBACK_REPO_OWNER}/{FEEDBACK_REPO_NAME}/git/trees/{FEEDBACK_REPO_BRANCH}?recursive=1"
        req = urllib.request.Request(api_url, headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "LM-Lab-Feedback",
        })

        with urllib.request.urlopen(req, timeout=15) as resp:
            tree_data = json.loads(resp.read().decode("utf-8"))

        # Filter files under data/feedback/
        feedback_files = [
            item for item in tree_data.get("tree", [])
            if item["type"] == "blob" and item["path"].startswith("data/feedback/")
        ]

        for file_item in feedback_files:
            try:
                file_path = file_item["path"]
                # Download file content
                content_url = f"https://api.github.com/repos/{FEEDBACK_REPO_OWNER}/{FEEDBACK_REPO_NAME}/contents/{file_path}?ref={FEEDBACK_REPO_BRANCH}"
                content_req = urllib.request.Request(content_url, headers={
                    "Authorization": f"Bearer {GITHUB_TOKEN}",
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "LM-Lab-Feedback",
                })

                with urllib.request.urlopen(content_req, timeout=10) as content_resp:
                    content_data = json.loads(content_resp.read().decode("utf-8"))

                file_content = base64.b64decode(content_data["content"])

                # Write to local filesystem
                local_path = Path(__file__).resolve().parent.parent.parent / file_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_bytes(file_content)
                synced += 1
            except Exception as e:
                errors.append({"file": file_path, "error": str(e)})

        return {
            "ok": True,
            "synced": synced,
            "total": len(feedback_files),
            "errors": errors[:10],  # Limit error list
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GitHub sync failed: {e}")


def _create_export_zip() -> io.BytesIO:
    """Helper para crear ZIP de export."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add active feedbacks
        for page_dir in FEEDBACK_DIR.iterdir():
            if not page_dir.is_dir():
                continue
            for section_dir in page_dir.iterdir():
                if not section_dir.is_dir():
                    continue
                for file in section_dir.iterdir():
                    if not file.is_file():
                        continue
                    arc_name = f"active/{page_dir.name}/{section_dir.name}/{file.name}"
                    zipf.write(file, arc_name)

        # Add archived feedbacks
        for page_dir in ARCHIVED_FEEDBACK_DIR.iterdir():
            if not page_dir.is_dir():
                continue
            for section_dir in page_dir.iterdir():
                if not section_dir.is_dir():
                    continue
                for file in section_dir.iterdir():
                    if not file.is_file():
                        continue
                    arc_name = f"archived/{page_dir.name}/{section_dir.name}/{file.name}"
                    zipf.write(file, arc_name)
    zip_buffer.seek(0)
    return zip_buffer


def _auto_export_daily():
    """Background task: auto-export feedbacks cada día a las 3am UTC."""
    try:
        import schedule  # type: ignore
    except Exception as e:
        print(f"⚠️ Auto-export disabled (missing optional dependency 'schedule'): {e}")
        return
    
    def do_export():
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            zip_path = BACKUPS_DIR / f"feedbacks_export_{timestamp}.zip"
            zip_buffer = _create_export_zip()
            zip_path.write_bytes(zip_buffer.read())
            print(f"✅ Auto-export completado: {zip_path}")
            
            # Cleanup: mantener solo últimos 7 backups
            backups = sorted(BACKUPS_DIR.glob("feedbacks_export_*.zip"), reverse=True)
            for old_backup in backups[7:]:
                old_backup.unlink()
                print(f"🗑️ Eliminado backup antiguo: {old_backup.name}")
        except Exception as e:
            print(f"❌ Auto-export failed: {e}")
    
    schedule.every().day.at("03:00").do(do_export)
    
    while True:
        schedule.run_pending()
        time.sleep(60)


# Iniciar auto-export thread
if os.getenv("FEEDBACK_AUTO_EXPORT", "0") == "1":
    _export_thread = threading.Thread(target=_auto_export_daily, daemon=True)
    _export_thread.start()
else:
    _export_thread = None


@router.get("/feedback/export")
async def export_all_feedbacks():
    """Export todos los feedbacks como ZIP (JSON + screenshots)."""
    zip_buffer = _create_export_zip()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        io.BytesIO(zip_buffer.read()),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=feedbacks_export_{timestamp}.zip"}
    )


@router.post("/feedback/ai-triage")
async def ai_triage_feedback(payload: dict):
    """AI auto-triage: clasifica feedback usando LM-Lab."""
    page_id = payload.get("page_id")
    section_id = payload.get("section_id")
    basename = payload.get("basename")
    
    if not all([page_id, section_id, basename]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    # Read feedback
    feedback_path = _resolve_feedback_path(page_id, section_id, basename, archived=payload.get("archived", False))
    if not feedback_path or not feedback_path.exists():
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    feedback_data = json.loads(feedback_path.read_text(encoding="utf-8"))
    title = feedback_data.get("title", "")
    comment = feedback_data.get("comment", "")
    feedback_type = feedback_data.get("feedback_type", "")
    
    # Heuristic triage (safe fallback). If you later expose a real LLM helper,
    # we can swap this to a model-backed implementation.
    suggestions = _ai_triage_heuristic(title=title, comment=comment, feedback_type=feedback_type)
    return {
        "ok": True,
        "suggestions": suggestions,
        "model": "heuristic",
    }


@router.get("/feedback/analytics")
async def feedback_analytics():
    """Analytics dashboard: estadísticas completas."""
    all_items = []
    
    # Active feedbacks
    for page_dir in FEEDBACK_DIR.iterdir():
        if not page_dir.is_dir():
            continue
        for section_dir in page_dir.iterdir():
            if not section_dir.is_dir():
                continue
            for json_file in section_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    data["archived"] = False
                    all_items.append(data)
                except:
                    pass
    
    # Archived feedbacks
    for page_dir in ARCHIVED_FEEDBACK_DIR.iterdir():
        if not page_dir.is_dir():
            continue
        for section_dir in page_dir.iterdir():
            if not section_dir.is_dir():
                continue
            for json_file in section_dir.glob("*.json"):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    data["archived"] = True
                    all_items.append(data)
                except:
                    pass
    
    # Calcular stats
    total = len(all_items)
    by_state = {}
    by_type = {}
    by_page = {}
    all_tags = []
    unread_count = 0
    
    for item in all_items:
        # State
        state = item.get("state", "new")
        by_state[state] = by_state.get(state, 0) + 1
        
        # Type
        ftype = item.get("feedback_type", "unknown")
        by_type[ftype] = by_type.get(ftype, 0) + 1
        
        # Page
        page = item.get("page_id", "unknown")
        by_page[page] = by_page.get(page, 0) + 1
        
        # Tags
        tags = item.get("tags", [])
        all_tags.extend(tags)
        
        # Unread
        if not item.get("read", False):
            unread_count += 1
    
    # Top tags
    from collections import Counter
    tag_counts = Counter(all_tags)
    top_tags = [{
        "tag": tag,
        "count": count
    } for tag, count in tag_counts.most_common(10)]
    
    return {
        "total": total,
        "unread": unread_count,
        "by_state": by_state,
        "by_type": by_type,
        "by_page": by_page,
        "top_tags": top_tags,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/feedback/dashboard", response_class=HTMLResponse)
async def feedback_analytics_dashboard():
    """Serve the analytics dashboard."""
    html_path = TEMPLATE_DIR / "feedback_analytics.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Analytics dashboard not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.get("/feedback/viewer", response_class=HTMLResponse)
async def feedback_viewer():
    """Serve the feedback viewer dashboard (localhost only)."""
    html_path = TEMPLATE_DIR / "feedback_viewer.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Viewer template not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
