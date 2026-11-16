"""Idempotent Google Drive v3 folder synchronizer using authenticated gcloud OAuth token."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import subprocess
import time
from typing import Dict, Optional

import requests

SKIP_DIRS = {".git", "__pycache__", "build", "pit_mem_core.egg-info", ".pytest_cache", ".venv"}
SKIP_SUFFIXES = {".so", ".o", ".tmp"}


def get_drive_token() -> str:
    out = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    if not out:
        raise RuntimeError("Empty gcloud access token")
    return out


def _get_with_retry(url: str, headers: Dict[str, str], params: Dict[str, str]) -> requests.Response:
    for attempt in range(5):
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code in (403, 429, 500, 502, 503, 504) and attempt < 4:
            time.sleep(1.5 * (2 ** attempt))
            headers = {"Authorization": f"Bearer {get_drive_token()}"}
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def ensure_drive_folder(name: str, parent_id: str, token: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    q = (
        f"name = '{name}' and '{parent_id}' in parents "
        f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    r = _get_with_retry(
        "https://www.googleapis.com/drive/v3/files",
        headers=headers,
        params={"q": q, "fields": "files(id, name)", "pageSize": 5},
    )
    files = r.json().get("files", [])
    if files:
        return files[0]["id"]

    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    r2 = requests.post(
        "https://www.googleapis.com/drive/v3/files",
        headers={**headers, "Content-Type": "application/json"},
        data=json.dumps(meta),
        params={"fields": "id"},
        timeout=20,
    )
    r2.raise_for_status()
    return r2.json()["id"]


def upload_or_update_file(local_path: str, parent_id: str, token: str) -> str:
    fname = os.path.basename(local_path)
    mime, _ = mimetypes.guess_type(local_path)
    if not mime:
        mime = "application/octet-stream"

    headers = {"Authorization": f"Bearer {token}"}
    q = f"name = '{fname}' and '{parent_id}' in parents and trashed = false"
    r = _get_with_retry(
        "https://www.googleapis.com/drive/v3/files",
        headers=headers,
        params={"q": q, "fields": "files(id, name, md5Checksum)", "pageSize": 5},
    )
    existing = r.json().get("files", [])
    if existing and existing[0].get("md5Checksum"):
        h = hashlib.md5()
        with open(local_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == existing[0]["md5Checksum"]:
            return existing[0]["id"]

    for attempt in range(5):
        file_size = os.path.getsize(local_path)
        if existing:
            fid = existing[0]["id"]
            if file_size < 5 * 1024 * 1024:
                with open(local_path, "rb") as f:
                    resp = requests.patch(
                        f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=media",
                        headers={**headers, "Content-Type": mime},
                        data=f.read(),
                        timeout=120,
                    )
                if resp.status_code in (403, 429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(1.5 * (2 ** attempt))
                    token = get_drive_token()
                    headers = {"Authorization": f"Bearer {token}"}
                    continue
                resp.raise_for_status()
                return resp.json().get("id", fid)
            init_url = f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=resumable"
            init_resp = requests.patch(
                init_url,
                headers={**headers, "Content-Type": "application/json; charset=UTF-8", "X-Upload-Content-Type": mime},
                data=json.dumps({"name": fname}),
                timeout=30,
            )
        else:
            init_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
            init_resp = requests.post(
                init_url,
                headers={**headers, "Content-Type": "application/json; charset=UTF-8", "X-Upload-Content-Type": mime},
                data=json.dumps({"name": fname, "parents": [parent_id]}),
                timeout=30,
            )
        if init_resp.status_code in (403, 429, 500, 502, 503, 504) and attempt < 4:
            time.sleep(1.5 * (2 ** attempt))
            token = get_drive_token()
            headers = {"Authorization": f"Bearer {token}"}
            continue
        init_resp.raise_for_status()
        session_uri = init_resp.headers["Location"]

        with open(local_path, "rb") as f:
            payload = f.read()
        put_resp = requests.put(
            session_uri,
            headers={**headers, "Content-Length": str(len(payload)), "Content-Type": mime},
            data=payload,
            timeout=300,
        )
        if put_resp.status_code in (403, 429, 500, 502, 503, 504) and attempt < 4:
            time.sleep(1.5 * (2 ** attempt))
            token = get_drive_token()
            headers = {"Authorization": f"Bearer {token}"}
            continue
        put_resp.raise_for_status()
        return put_resp.json().get("id", "")
    raise RuntimeError(f"Failed to upload {local_path} after 5 attempts")


def sync_directory_to_gdrive(local_root: str, drive_folder_name: str = "PiT-Mem_VLDB_Project") -> str:
    token = get_drive_token()
    root_id = ensure_drive_folder(drive_folder_name, "root", token)
    # Ensure public "Anyone with the link can view" permission for PVLDB Artifact Availability
    requests.post(
        f"https://www.googleapis.com/drive/v3/files/{root_id}/permissions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        data=json.dumps({"type": "anyone", "role": "reader"}),
        timeout=20,
    )
    folder_map: Dict[str, str] = {os.path.abspath(local_root): root_id}

    uploaded_count = 0
    for dirpath, dirnames, filenames in os.walk(local_root):
        dirnames[:] = sorted([d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")])
        abs_dir = os.path.abspath(dirpath)
        parent_drive_id = folder_map[abs_dir]

        # Refresh token periodically on large trees
        token = get_drive_token()
        for d in dirnames:
            child_abs = os.path.join(abs_dir, d)
            folder_map[child_abs] = ensure_drive_folder(d, parent_drive_id, token)

        for fn in sorted(filenames):
            if fn.startswith(".") or any(fn.endswith(suf) for suf in SKIP_SUFFIXES):
                continue
            fpath = os.path.join(abs_dir, fn)
            upload_or_update_file(fpath, parent_drive_id, token)
            uploaded_count += 1

    url = f"https://drive.google.com/drive/folders/{root_id}"
    print(f"[Google Drive Sync] Uploaded/updated {uploaded_count} files -> {url}")
    return url


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "."
    name = sys.argv[2] if len(sys.argv) > 2 else "PiT-Mem_VLDB_Project"
    sync_directory_to_gdrive(src, name)
