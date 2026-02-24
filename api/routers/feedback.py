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
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["feedback"])

# ---------------------------------------------------------------------------
#  Storage — organized as data/feedback/{page_id}/{section_id}/{timestamp}.*
# ---------------------------------------------------------------------------

FEEDBACK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVED_FEEDBACK_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "feedback_archived"
ARCHIVED_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

# ---------------------------------------------------------------------------
#  In-memory rate limit: 1 per IP+section per 30 seconds
# ---------------------------------------------------------------------------

_rate_limit: dict[str, float] = {}
RATE_LIMIT_SECONDS = 30  # 30 seconds


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


def _check_rate_limit(ip: str, page_id: str, section_id: str) -> None:
    key = f"{ip}:{page_id}:{section_id}"
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


class FeedbackResponse(BaseModel):
    ok: bool
    filename: str


class FeedbackActionPayload(BaseModel):
    page_id: str = Field(..., min_length=1, max_length=64)
    section_id: str = Field(..., min_length=1, max_length=64)
    basename: str = Field(..., min_length=1, max_length=64)
    archived: bool = False


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
                "timestamp": data.get("timestamp"),
                "title": data.get("title"),
                "comment": data.get("comment"),
                "name": data.get("name"),
                "anon_id": data.get("anon_id"),
                "github_issue_url": data.get("github_issue_url"),
                "has_screenshot": screenshot_name is not None,
                "screenshot_url": screenshot_url,
                "has_user_screenshot": user_screenshot_name is not None,
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
    _check_rate_limit(client_ip, payload.page_id, payload.section_id)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_page = _safe_name(payload.page_id)
    safe_section = _safe_name(payload.section_id)

    # Create nested directory: feedback/{page}/{section}/
    section_dir = FEEDBACK_DIR / safe_page / safe_section
    section_dir.mkdir(parents=True, exist_ok=True)

    basename = ts

    # Build JSON data (exclude screenshot bytes from stored JSON)
    data = {
        "timestamp": ts,
        "page_id": payload.page_id,
        "section_id": payload.section_id,
        "pinned": False,
        "title": payload.title,
        "comment": payload.comment,
        "anon_id": payload.anon_id,
        "name": payload.name,
        "has_screenshot": False,
        "has_user_screenshot": payload.user_screenshot_b64 is not None,
        "read": False,
        "read_at": None,
        "github_issue_url": None,
        "ip": client_ip,
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

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
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
            "Authorization": f"Bearer {github_token}",
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


@router.get("/feedback/viewer", response_class=HTMLResponse)
async def feedback_viewer():
    """Serve the feedback viewer dashboard (localhost only)."""
    html_path = TEMPLATE_DIR / "feedback_viewer.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Viewer template not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
