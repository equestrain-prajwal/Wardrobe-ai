"""
Wardrobe AI
===========
Module 1 - App shell, screen navigation and the local storage layer.
Module 2 - Scanner: premium UI, native dark mode, offline vision engine,
           optional cloud cut-outs and one-tap confirmation badge.

Target stack (pinned):
    Kivy   2.3.x
    KivyMD 1.2.0   (strict - the app refuses to start on any other version)
    OpenCV 4.x + NumPy   (vision engine; the app still runs without them,
                          the Scanner just reports that they are missing)

Everything is stored privately inside ``App.user_data_dir``:
    Android : /data/data/<package>/files
    Windows : %APPDATA%\\wardrobe
    Linux   : ~/.config/wardrobe
    macOS   : ~/Library/Application Support/wardrobe

Background removal, in order of preference:
    1. Optional cloud provider (OFF by default - photos leave the device only
       if the user picks remove.bg / Photoroom / a self-hosted rembg server
       in Profile > Background removal and supplies a key or URL).
    2. On-device U2-Net-p (assets/models/u2netp.onnx) via OpenCV DNN.
    3. On-device colour key / GrabCut (no model needed).
Any cloud failure or timeout silently falls back to on-device processing.

Offline AI models live in ``assets/models/`` next to this file. Run
``python tools/prepare_models.py --help`` on a PC to fetch/export them.

Later modules should only touch disk through ``StorageManager`` (or the
module-level ``atomic_write_json`` / ``atomic_write_bytes`` / ``read_json`` /
``update_json`` helpers), never with a bare ``open(..., "w")``.
"""

from __future__ import annotations

import colorsys
import copy
import functools
import json
import math
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Desktop window sizing - must run BEFORE kivy.core.window is imported
# (kivymd.app imports it), otherwise the Config values are ignored.
# ---------------------------------------------------------------------------
from kivy.config import Config
from kivy.utils import platform

IS_ANDROID = platform == "android"
IS_MOBILE = platform in ("android", "ios")

if not IS_MOBILE:
    # Approximate a modern phone viewport while developing on desktop.
    Config.set("graphics", "width", "412")
    Config.set("graphics", "height", "860")
    Config.set("graphics", "resizable", "1")
    # Disable the red multitouch-emulation dots on right click.
    Config.set("input", "mouse", "mouse,multitouch_on_demand")

import kivymd  # noqa: E402

KIVYMD_REQUIRED = "1.2.0"
if kivymd.__version__ != KIVYMD_REQUIRED:
    raise RuntimeError(
        f"Wardrobe AI is pinned to KivyMD {KIVYMD_REQUIRED}, but "
        f"{kivymd.__version__} is installed. Run: pip install kivymd=={KIVYMD_REQUIRED}"
    )

from kivy.animation import Animation  # noqa: E402
from kivy.clock import Clock  # noqa: E402
from kivy.core.window import Window  # noqa: E402
from kivy.graphics import Color, Ellipse, Line, PopMatrix, PushMatrix, Rotate  # noqa: E402
from kivy.lang import Builder  # noqa: E402
from kivy.logger import Logger  # noqa: E402
from kivy.metrics import dp  # noqa: E402
from kivy.properties import (  # noqa: E402
    BooleanProperty,
    ColorProperty,
    DictProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
)
from kivy.uix.behaviors import ButtonBehavior  # noqa: E402
from kivy.uix.boxlayout import BoxLayout  # noqa: E402
from kivy.uix.label import Label  # noqa: E402
from kivy.uix.relativelayout import RelativeLayout  # noqa: E402
from kivy.uix.widget import Widget  # noqa: E402
from kivy.uix.modalview import ModalView  # noqa: E402
from kivy.uix.screenmanager import FadeTransition  # noqa: E402
from kivy.utils import get_color_from_hex  # noqa: E402
from kivymd.app import MDApp  # noqa: E402
from kivymd.uix.boxlayout import MDBoxLayout  # noqa: E402
from kivymd.uix.button import MDFlatButton  # noqa: E402
from kivymd.uix.card import MDCard  # noqa: E402
from kivymd.uix.dialog import MDDialog  # noqa: E402
from kivymd.uix.filemanager import MDFileManager  # noqa: E402
from kivymd.uix.floatlayout import MDFloatLayout  # noqa: E402
from kivymd.uix.relativelayout import MDRelativeLayout  # noqa: E402
from kivymd.uix.screen import MDScreen  # noqa: E402
from kivymd.uix.screenmanager import MDScreenManager  # noqa: E402

# Vision dependencies are optional at import time so the shell always starts.
try:
    import cv2
    import numpy as np

    # OpenCL initialisation costs ~1.7 s on the first OpenCV call on some GPU
    # drivers, and our phone-sized images run faster on the CPU anyway.
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:  # noqa: BLE001 - builds without the ocl module
        pass

    CV_AVAILABLE = True
    CV_IMPORT_ERROR = ""
except Exception as _cv_exc:  # pragma: no cover - depends on environment
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    CV_AVAILABLE = False
    CV_IMPORT_ERROR = str(_cv_exc)

APP_VERSION = "0.3.0"
SCHEMA_VERSION = 1
LOG_TAG = "WardrobeAI"

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "assets", "models")


# ===========================================================================
# STORAGE LAYER
# ===========================================================================

#: Top-level data folders created inside user_data_dir.
DATA_DIRS: Tuple[str, ...] = (
    "wardrobe",
    "profile",
    "cache",
    "logs",
    "outfits",
    "wishlist",
    "packing",
)

#: Image sub-folders. Paths stored in JSON use forward slashes and are
#: relative to user_data_dir so the data folder stays portable.
IMG_ORIGINAL_DIR = "wardrobe/images/original"
IMG_CUTOUT_DIR = "wardrobe/images/cutout"
IMG_THUMB_DIR = "wardrobe/images/thumb"

DATA_SUBDIRS: Tuple[str, ...] = (
    IMG_ORIGINAL_DIR,
    IMG_CUTOUT_DIR,
    IMG_THUMB_DIR,
    "profile/images",
)

#: Prefix of in-flight temp files; anything left with this prefix is debris
#: from a crash mid-write and is safe to delete at startup.
TMP_PREFIX = ".tmp-"


class StorageError(Exception):
    """Raised when the storage layer cannot complete a read/write safely."""


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Per-file locks --------------------------------------------------------
# Background threads (the scanner worker, weather/trend sync later) and the UI
# thread may touch the same file. One re-entrant lock per absolute path
# serialises read-modify-write cycles inside this process.
_path_locks: Dict[str, threading.RLock] = {}
_path_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(path))
    with _path_locks_guard:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _path_locks[key] = lock
        return lock


def _fsync_dir(directory: str) -> None:
    """Persist the directory entry after a rename (POSIX only; no-op on Windows)."""
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(
    path: str,
    payload: bytes,
    *,
    retries: int = 5,
    retry_delay: float = 0.05,
) -> None:
    """Write raw bytes to ``path`` atomically.

    Sequence:
      1. Write to a temp file in the SAME directory (same filesystem, so the
         rename is atomic), flush and fsync it.
      2. ``os.replace`` the temp file over the target. Readers see either the
         complete old file or the complete new file, never a partial one.
      3. fsync the directory so the rename survives a power loss (POSIX).

    On Windows, ``os.replace`` can briefly fail with PermissionError while an
    antivirus/indexer holds the target open, so it is retried with backoff.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    suffix = os.path.splitext(path)[1] or ".bin"

    with _lock_for(path):
        fd, tmp_path = tempfile.mkstemp(prefix=TMP_PREFIX, suffix=suffix, dir=directory)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())

            last_exc: Optional[BaseException] = None
            for attempt in range(retries):
                try:
                    os.replace(tmp_path, path)
                    last_exc = None
                    break
                except PermissionError as exc:
                    last_exc = exc
                    time.sleep(retry_delay * (attempt + 1))
            if last_exc is not None:
                raise StorageError(f"Could not replace {path}: {last_exc}") from last_exc

            _fsync_dir(directory)
        finally:
            # After a successful replace the temp file no longer exists;
            # after any failure, clean it up.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def atomic_write_json(path: str, data: Any, *, indent: Optional[int] = 2) -> None:
    """Serialise ``data`` as UTF-8 JSON and write it atomically.

    Serialisation happens in memory first, so a non-serialisable payload
    never touches disk.
    """
    try:
        payload = json.dumps(data, ensure_ascii=False, indent=indent)
    except (TypeError, ValueError) as exc:
        raise StorageError(f"Data for {path} is not JSON-serialisable: {exc}") from exc
    atomic_write_bytes(path, payload.encode("utf-8"))


def read_json(path: str, default: Any = None) -> Any:
    """Read JSON from ``path``.

    * Missing file  -> a deep copy of ``default``.
    * Corrupt file  -> the file is moved aside to ``<name>.corrupt-<timestamp>``
                       (kept for recovery, never silently deleted) and a deep
                       copy of ``default`` is returned.
    """
    with _lock_for(path):
        if not os.path.exists(path):
            return copy.deepcopy(default)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            quarantine = f"{path}.corrupt-{int(time.time())}"
            try:
                os.replace(path, quarantine)
                Logger.warning(f"{LOG_TAG}: corrupt JSON {path} moved to {quarantine} ({exc})")
            except OSError as move_exc:
                Logger.error(f"{LOG_TAG}: corrupt JSON {path} could not be moved: {move_exc}")
            return copy.deepcopy(default)


def update_json(
    path: str,
    mutator: Callable[[Any], Any],
    default: Any = None,
) -> Any:
    """Locked read-modify-write cycle.

    ``mutator`` receives the current data and may either modify it in place
    (return ``None``) or return a replacement object. The result is written
    atomically and returned.
    """
    with _lock_for(path):
        data = read_json(path, default)
        result = mutator(data)
        if result is not None:
            data = result
        atomic_write_json(path, data)
        return data


def deep_merge_defaults(defaults: Any, current: Any) -> Any:
    """Fill keys missing from ``current`` using ``defaults``, recursively.

    Existing user values are never overwritten. Used to upgrade
    settings.json when a newer app version adds new keys.
    """
    if not isinstance(defaults, dict) or not isinstance(current, dict):
        return copy.deepcopy(current)
    merged = copy.deepcopy(current)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = deep_merge_defaults(value, merged[key])
    return merged


# --- Default document factories -------------------------------------------
# Functions (not constants) so every call returns a fresh object with a
# current timestamp.

def default_settings() -> Dict[str, Any]:
    now = utc_now_iso()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "app": {
            "version": APP_VERSION,
            "first_run_complete": False,
            "last_launch": None,
        },
        # Network features are opt-in: the app is fully offline by default.
        "network": {
            "trends_enabled": False,
            "weather_enabled": False,
            "wifi_only": True,
        },
        "units": {"temperature": "C", "length": "cm"},
        "currency": "USD",
        "shuffle": {"shake_enabled": True, "shake_threshold": 2.5},
        "theme": {"style": "Light", "primary_palette": "Teal", "accent_palette": "Amber"},
        "privacy": {"delete_source_photos": False},
        "scanner": {
            # Below this confidence the badge invites the user to double-check.
            "confirm_threshold": 0.75,
        },
        "camera": {
            # Preview rotation in degrees clockwise (0/90/180/270). Android
            # back cameras are usually mounted sideways, hence 90 there.
            "rotation": 90 if IS_ANDROID else 0,
        },
        # Background-removal engine. "local" never uploads anything.
        "cutout": {
            "provider": "local",          # local | removebg | photoroom | rembg_server
            "removebg_api_key": "",
            "removebg_size": "preview",   # free-tier size on remove.bg
            "photoroom_api_key": "",
            "server_url": "",             # e.g. http://192.168.1.20:7000 (rembg s)
            "timeout_s": 8.0,
        },
    }


def default_items() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": utc_now_iso(), "items": {}}


def default_profile() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now_iso(),
        "source_photo": None,
        "body": {},
        "skin": {},
        "palette": {},
        "preferences": {"style_tags": [], "disliked_colors": [], "fit": "regular"},
        "location": {"lat": None, "lon": None, "city": None, "manual": False},
        "analysis": {},
    }


def default_trends() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now_iso(),
        "last_sync": None,
        "ttl_hours": 24,
        "sources": [],
        "trends": [],
        "matches": [],
    }


def default_wear_log() -> Dict[str, Any]:
    # entries: {"YYYY-MM-DD": [ {outfit_id, item_ids, occasion, weather_snapshot, rating}, ... ]}
    return {"schema_version": SCHEMA_VERSION, "updated_at": utc_now_iso(), "entries": {}}


def default_outfits() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": utc_now_iso(), "outfits": {}}


def default_wishlist() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": utc_now_iso(), "wishes": []}


def default_trips() -> Dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "updated_at": utc_now_iso(), "trips": []}


#: Logical document name -> (path relative to user_data_dir, default factory)
DOCUMENTS: Dict[str, Tuple[str, Callable[[], Dict[str, Any]]]] = {
    "settings": ("settings.json", default_settings),
    "items": ("wardrobe/items.json", default_items),
    "profile": ("profile/profile.json", default_profile),
    "trends": ("cache/trends_cache.json", default_trends),
    "wear_log": ("logs/wear_log.json", default_wear_log),
    "outfits": ("outfits/outfits.json", default_outfits),
    "wishlist": ("wishlist/wishlist.json", default_wishlist),
    "trips": ("packing/trips.json", default_trips),
}

_MISSING = object()


class StorageManager:
    """Single entry point for all on-device persistence.

    Usage from any module::

        storage = MDApp.get_running_app().storage
        items = storage.load("items")
        storage.update("items", lambda d: d["items"].update({item_id: item}))
        storage.set_setting("network.weather_enabled", True)
    """

    def __init__(self, root_dir: str) -> None:
        self.root = os.path.abspath(root_dir)

    # -- paths ------------------------------------------------------------
    def path(self, *parts: str) -> str:
        """Absolute path inside the data root. Accepts 'a/b' style parts.

        Rejects anything that would escape the root via '..'.
        """
        pieces: List[str] = []
        for part in parts:
            pieces.extend(p for p in part.replace("\\", "/").split("/") if p)
        full = os.path.abspath(os.path.join(self.root, *pieces))
        if os.path.commonpath([full, self.root]) != self.root:
            raise StorageError(f"Path escapes storage root: {parts!r}")
        return full

    def doc_path(self, name: str) -> str:
        if name not in DOCUMENTS:
            raise StorageError(f"Unknown document '{name}'. Known: {', '.join(DOCUMENTS)}")
        return self.path(DOCUMENTS[name][0])

    # -- lifecycle --------------------------------------------------------
    def initialize(self) -> bool:
        """Create folders and seed documents. Safe to call on every launch.

        Returns True if this is the very first launch (settings.json did not
        exist before this call).
        """
        try:
            os.makedirs(self.root, exist_ok=True)
            for rel in DATA_DIRS + DATA_SUBDIRS:
                os.makedirs(self.path(rel), exist_ok=True)
        except OSError as exc:
            raise StorageError(f"Cannot create data folders under {self.root}: {exc}") from exc

        self._purge_stale_temp_files()

        settings_path = self.doc_path("settings")
        first_launch = not os.path.exists(settings_path)

        # settings.json: create, or upgrade in place by filling new keys.
        current = read_json(settings_path, None)
        if not isinstance(current, dict):
            atomic_write_json(settings_path, default_settings())
        else:
            merged = deep_merge_defaults(default_settings(), current)
            merged["app"]["version"] = APP_VERSION
            if merged != current:
                merged["updated_at"] = utc_now_iso()
                atomic_write_json(settings_path, merged)

        # Every other document: create only if missing or unreadable.
        for name, (_rel, factory) in DOCUMENTS.items():
            if name == "settings":
                continue
            doc_path = self.doc_path(name)
            if not isinstance(read_json(doc_path, None), dict):
                atomic_write_json(doc_path, factory())

        Logger.info(f"{LOG_TAG}: storage ready at {self.root} (first_launch={first_launch})")
        return first_launch

    def _purge_stale_temp_files(self) -> None:
        """Delete temp files left behind by a crash during a previous write."""
        folders = [self.root] + [self.path(rel) for rel in DATA_DIRS + DATA_SUBDIRS]
        for folder in folders:
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            for fname in names:
                if fname.startswith(TMP_PREFIX):
                    try:
                        os.remove(os.path.join(folder, fname))
                        Logger.info(f"{LOG_TAG}: removed stale temp file {fname}")
                    except OSError:
                        pass

    def purge_orphan_images(self, min_age_seconds: float = 3600.0) -> int:
        """Delete garment images that no saved item references.

        Scans produce image files before the user confirms; if the app is
        killed on the review screen those files would otherwise leak. Only
        files older than ``min_age_seconds`` are touched, so a scan that is in
        progress right now is never affected. Also clears old scanner input
        files from cache/.
        """
        items = self.load("items").get("items", {})
        known_ids = set(items.keys()) if isinstance(items, dict) else set()
        now = time.time()
        removed = 0

        for rel_dir in (IMG_ORIGINAL_DIR, IMG_CUTOUT_DIR, IMG_THUMB_DIR):
            folder = self.path(rel_dir)
            try:
                names = os.listdir(folder)
            except OSError:
                continue
            for fname in names:
                item_id = fname.split("_", 1)[0]
                full = os.path.join(folder, fname)
                try:
                    if item_id not in known_ids and now - os.path.getmtime(full) > min_age_seconds:
                        os.remove(full)
                        removed += 1
                except OSError:
                    pass

        cache_dir = self.path("cache")
        try:
            for fname in os.listdir(cache_dir):
                if fname.startswith(("capture_", "pick_", "session_")):
                    full = os.path.join(cache_dir, fname)
                    try:
                        if now - os.path.getmtime(full) > min_age_seconds:
                            os.remove(full)
                            removed += 1
                    except OSError:
                        pass
        except OSError:
            pass

        if removed:
            Logger.info(f"{LOG_TAG}: purged {removed} orphan image file(s)")
        return removed

    # -- documents --------------------------------------------------------
    def load(self, name: str) -> Dict[str, Any]:
        factory = DOCUMENTS[name][1] if name in DOCUMENTS else dict
        data = read_json(self.doc_path(name), None)
        return data if isinstance(data, dict) else factory()

    def save(self, name: str, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise StorageError(f"Document '{name}' must be a dict, got {type(data).__name__}")
        data["updated_at"] = utc_now_iso()
        data.setdefault("schema_version", SCHEMA_VERSION)
        atomic_write_json(self.doc_path(name), data)

    def update(self, name: str, mutator: Callable[[Dict[str, Any]], Any]) -> Dict[str, Any]:
        """Locked read-modify-write of a named document."""
        path = self.doc_path(name)
        with _lock_for(path):
            data = self.load(name)
            result = mutator(data)
            if result is not None:
                data = result
            self.save(name, data)
            return data

    # -- settings shortcuts (dotted keys, e.g. "theme.style") --------------
    def get_setting(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self.load("settings")
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_setting(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")

        def _apply(data: Dict[str, Any]) -> None:
            node = data
            for part in parts[:-1]:
                child = node.get(part, _MISSING)
                if not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
            node[parts[-1]] = value

        self.update("settings", _apply)

    # -- diagnostics ------------------------------------------------------
    def status(self) -> List[Tuple[str, bool, int]]:
        """[(relative_path, exists, size_bytes), ...] for every document."""
        rows = []
        for _name, (rel, _factory) in DOCUMENTS.items():
            full = self.path(rel)
            exists = os.path.isfile(full)
            rows.append((rel, exists, os.path.getsize(full) if exists else 0))
        return rows

    def self_test(self) -> Tuple[bool, str]:
        """Round-trip an atomic write/read and check that no temp files leak."""
        probe = self.path("cache", "_selftest.json")
        payload = {
            "token": uuid.uuid4().hex,
            "unicode": "héllo wörld",
            "nested": {"numbers": [1, 2, 3], "flag": True, "none": None},
            "written_at": utc_now_iso(),
        }
        started = time.perf_counter()
        try:
            atomic_write_json(probe, payload)
            readback = read_json(probe, None)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            leftovers = [
                f for f in os.listdir(os.path.dirname(probe)) if f.startswith(TMP_PREFIX)
            ]
            if readback != payload:
                return False, "Read-back data did not match what was written."
            if leftovers:
                return False, f"Temp files leaked: {', '.join(leftovers)}"
            return True, f"Atomic write + read OK in {elapsed_ms:.1f} ms."
        except (OSError, StorageError) as exc:
            return False, f"Storage error: {exc}"
        finally:
            try:
                if os.path.exists(probe):
                    os.remove(probe)
            except OSError:
                pass


# ===========================================================================
# GARMENT TAXONOMY
# ===========================================================================

CATEGORIES: Tuple[str, ...] = (
    "T-Shirt",
    "Shirt",
    "Sweater",
    "Hoodie",
    "Jacket",
    "Coat",
    "Pants",     # casual: chinos, cargos, joggers
    "Trouser",   # tailored / formal
    "Jeans",
    "Shorts",
    "Skirt",
    "Dress",
    "Shoes",
    "Accessory",
)

#: Static defaults per category. layer_level only applies to the "top" slot:
#: 0 = base (T-shirt), 1 = mid (shirt/sweater), 2 = insulating (hoodie),
#: 3 = outer (jacket/coat). Other slots use None.
CATEGORY_META: Dict[str, Dict[str, Any]] = {
    "T-Shirt": {"slot": "top", "layer_level": 0, "warmth": 1, "formality": 1,
                "occasions": ["Casual", "Active", "DateNight"],
                "seasons": ["spring", "summer", "autumn"]},
    "Shirt": {"slot": "top", "layer_level": 1, "warmth": 2, "formality": 3,
              "occasions": ["Casual", "Formal", "DateNight"],
              "seasons": ["spring", "summer", "autumn", "winter"]},
    "Sweater": {"slot": "top", "layer_level": 1, "warmth": 3, "formality": 2,
                "occasions": ["Casual", "DateNight", "Winter"],
                "seasons": ["autumn", "winter"]},
    "Hoodie": {"slot": "top", "layer_level": 2, "warmth": 3, "formality": 1,
               "occasions": ["Casual", "Active", "Winter"],
               "seasons": ["spring", "autumn", "winter"]},
    "Jacket": {"slot": "top", "layer_level": 3, "warmth": 3, "formality": 3,
               "occasions": ["Casual", "Formal", "DateNight", "Winter"],
               "seasons": ["spring", "autumn", "winter"]},
    "Coat": {"slot": "top", "layer_level": 3, "warmth": 5, "formality": 4,
             "occasions": ["Formal", "Casual", "Winter"],
             "seasons": ["autumn", "winter"]},
    "Pants": {"slot": "bottom", "layer_level": None, "warmth": 2, "formality": 2,
              "occasions": ["Casual", "DateNight"],
              "seasons": ["spring", "summer", "autumn", "winter"]},
    "Trouser": {"slot": "bottom", "layer_level": None, "warmth": 2, "formality": 4,
                "occasions": ["Formal", "DateNight"],
                "seasons": ["spring", "summer", "autumn", "winter"]},
    "Jeans": {"slot": "bottom", "layer_level": None, "warmth": 2, "formality": 2,
              "occasions": ["Casual", "DateNight"],
              "seasons": ["spring", "autumn", "winter"]},
    "Shorts": {"slot": "bottom", "layer_level": None, "warmth": 1, "formality": 1,
               "occasions": ["Casual", "Active"],
               "seasons": ["summer"]},
    "Skirt": {"slot": "bottom", "layer_level": None, "warmth": 1, "formality": 3,
              "occasions": ["Casual", "DateNight", "Formal"],
              "seasons": ["spring", "summer", "autumn"]},
    "Dress": {"slot": "full_body", "layer_level": None, "warmth": 1, "formality": 4,
              "occasions": ["DateNight", "Formal", "Casual"],
              "seasons": ["spring", "summer", "autumn"]},
    "Shoes": {"slot": "footwear", "layer_level": None, "warmth": 2, "formality": 2,
              "occasions": ["Casual", "Formal", "DateNight", "Active", "Winter"],
              "seasons": ["spring", "summer", "autumn", "winter"]},
    "Accessory": {"slot": "accessory", "layer_level": None, "warmth": 1, "formality": 2,
                  "occasions": ["Casual", "Formal", "DateNight"],
                  "seasons": ["spring", "summer", "autumn", "winter"]},
}

#: ImageNet-1k class index -> weighted garment categories. ImageNet has no
#: generic "shirt" or "trousers" class, so those rely on the silhouette
#: analyser plus the one-tap confirmation.
IMAGENET_CATEGORY_MAP: Dict[int, Dict[str, float]] = {
    610: {"T-Shirt": 1.0},                    # jersey, T-shirt
    841: {"Hoodie": 0.65, "Sweater": 0.35},   # sweatshirt
    474: {"Sweater": 1.0},                    # cardigan
    869: {"Coat": 1.0},                       # trench coat
    568: {"Coat": 1.0},                       # fur coat
    501: {"Coat": 1.0},                       # cloak
    735: {"Coat": 0.6, "Jacket": 0.4},        # poncho
    617: {"Jacket": 0.5, "Coat": 0.5},        # lab coat
    834: {"Jacket": 0.75, "Trouser": 0.25},   # suit
    608: {"Jeans": 1.0},                      # jean, blue jean, denim
    655: {"Skirt": 1.0},                      # miniskirt
    689: {"Skirt": 1.0},                      # overskirt
    601: {"Skirt": 1.0},                      # hoopskirt
    775: {"Skirt": 1.0},                      # sarong
    578: {"Dress": 1.0},                      # gown
    399: {"Dress": 1.0},                      # abaya
    614: {"Dress": 0.6, "Coat": 0.4},         # kimono
    842: {"Shorts": 1.0},                     # swimming trunks
    697: {"Pants": 0.5, "Shirt": 0.5},        # pajama
    770: {"Shoes": 1.0},                      # running shoe
    514: {"Shoes": 1.0},                      # cowboy boot
    630: {"Shoes": 1.0},                      # loafer
    774: {"Shoes": 1.0},                      # sandal
    502: {"Shoes": 1.0},                      # clog
    457: {"Accessory": 1.0},                  # bow tie
    806: {"Accessory": 1.0},                  # sock
    824: {"Accessory": 1.0},                  # stole
    414: {"Accessory": 1.0},                  # backpack
    748: {"Accessory": 1.0},                  # purse
    893: {"Accessory": 1.0},                  # wallet
    837: {"Accessory": 1.0},                  # sunglasses
    515: {"Accessory": 1.0},                  # cowboy hat
    658: {"Accessory": 1.0},                  # mitten
}

#: Consumer-friendly fashion colour names (name, hex), matched in CIELAB.
NAMED_COLORS: Tuple[Tuple[str, str], ...] = (
    ("Jet Black", "#111111"), ("Charcoal Grey", "#36454F"), ("Graphite Grey", "#555555"),
    ("Heather Grey", "#808080"), ("Ash Grey", "#B8B8B8"), ("Optic White", "#F8F8F8"),
    ("Ivory", "#FFFFF0"), ("Cream", "#F3E5AB"), ("Sand Beige", "#D9C3A0"),
    ("Khaki", "#C3B091"), ("Camel Brown", "#C19A6B"), ("Tan", "#D2B48C"),
    ("Mocha Brown", "#6F4E37"), ("Dark Chocolate", "#3E2723"), ("Rust Orange", "#B7410E"),
    ("Terracotta", "#E2725B"), ("Bright Orange", "#F28C28"), ("Mustard Yellow", "#E1AD01"),
    ("Sunflower Yellow", "#F5D33F"), ("Olive Green", "#6B6E2F"), ("Army Green", "#4B5320"),
    ("Sage Green", "#9CAF88"), ("Mint Green", "#98D8C8"), ("Emerald Green", "#2E8B57"),
    ("Forest Green", "#1F4D2B"), ("Teal", "#008080"), ("Turquoise", "#40E0D0"),
    ("Sky Blue", "#87CEEB"), ("Powder Blue", "#ADD8E6"), ("Denim Blue", "#3B5B92"),
    ("Royal Blue", "#2B4FBF"), ("Navy Blue", "#1F2A44"), ("Indigo", "#3F3F7F"),
    ("Royal Purple", "#6A3D9A"), ("Lavender", "#B57EDC"), ("Plum", "#8E4585"),
    ("Burgundy", "#800020"), ("Crimson Red", "#C62828"), ("Coral", "#FF7F50"),
    ("Baby Pink", "#F4A6C0"), ("Hot Pink", "#E5398A"), ("Dusty Rose", "#E8C4C0"),
    ("Antique Gold", "#C8A951"),
)

NEUTRAL_COLOR_NAMES = frozenset({
    "Jet Black", "Charcoal Grey", "Graphite Grey", "Heather Grey", "Ash Grey",
    "Optic White", "Ivory", "Cream", "Sand Beige", "Navy Blue",
})
EARTH_COLOR_NAMES = frozenset({
    "Khaki", "Camel Brown", "Tan", "Mocha Brown", "Dark Chocolate", "Rust Orange",
    "Terracotta", "Olive Green", "Army Green", "Mustard Yellow", "Sage Green",
})

IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

#: Categories phrased in the plural ("Are these Jeans?").
PLURAL_CATEGORIES = frozenset({"Pants", "Trouser", "Jeans", "Shorts", "Shoes"})


def layers_under(category: str) -> List[int]:
    """Layer levels an item of ``category`` may be worn over."""
    meta = CATEGORY_META.get(category, {})
    level = meta.get("layer_level")
    if meta.get("slot") != "top" or level is None:
        return []
    return list(range(level))


def guess_fabric_tags(category: str, pattern: str) -> List[str]:
    tags: List[str] = []
    if category == "Jeans":
        tags.append("denim")
    elif category == "Sweater":
        tags.append("knit")
    elif category == "Hoodie":
        tags.append("fleece")
    elif category == "T-Shirt":
        tags.append("jersey")
    if pattern == "printed":
        tags.append("printed")
    return tags


# ===========================================================================
# OPTIONAL CLOUD CUT-OUT PROVIDERS (opt-in; never used unless configured)
# ===========================================================================

CLOUD_PROVIDERS: Dict[str, str] = {
    "local": "On-device",
    "removebg": "remove.bg",
    "photoroom": "Photoroom",
    "rembg_server": "Self-hosted rembg",
}


class CloudCutoutError(Exception):
    """The cloud provider could not deliver a usable cut-out."""


def _encode_multipart(fields: Dict[str, str],
                      files: Dict[str, Tuple[str, bytes, str]]) -> Tuple[bytes, str]:
    boundary = "----WardrobeAI" + uuid.uuid4().hex
    parts: List[bytes] = []
    for name, value in fields.items():
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode("utf-8"),
            b"\r\n",
        ]
    for name, (filename, data, content_type) in files.items():
        parts += [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            data,
            b"\r\n",
        ]
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _ssl_context():
    import ssl
    try:  # Android's Python often lacks a CA bundle; certifi fixes that if bundled.
        import certifi  # type: ignore[import-not-found]
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def cloud_provider_ready(cfg: Dict[str, Any]) -> bool:
    provider = cfg.get("provider", "local")
    if provider == "removebg":
        return bool(str(cfg.get("removebg_api_key", "")).strip())
    if provider == "photoroom":
        return bool(str(cfg.get("photoroom_api_key", "")).strip())
    if provider == "rembg_server":
        return str(cfg.get("server_url", "")).strip().startswith(("http://", "https://"))
    return False


def cloud_cutout(provider: str, jpeg: bytes, cfg: Dict[str, Any]) -> bytes:
    """Upload ``jpeg`` to the chosen provider and return its PNG cut-out.

    Raises CloudCutoutError on any failure so the caller can fall back to
    on-device processing. Only a generic User-Agent is sent - no device or
    user identifiers.
    """
    import urllib.error
    import urllib.request

    timeout = max(2.0, float(cfg.get("timeout_s", 8.0)))
    headers = {"Accept": "image/png", "User-Agent": f"WardrobeAI/{APP_VERSION}"}
    fields: Dict[str, str] = {}

    if provider == "removebg":
        key = str(cfg.get("removebg_api_key", "")).strip()
        if not key:
            raise CloudCutoutError("no remove.bg API key set")
        url = "https://api.remove.bg/v1.0/removebg"
        headers["X-Api-Key"] = key
        fields = {"size": str(cfg.get("removebg_size") or "preview"), "format": "png"}
        files = {"image_file": ("garment.jpg", jpeg, "image/jpeg")}
    elif provider == "photoroom":
        key = str(cfg.get("photoroom_api_key", "")).strip()
        if not key:
            raise CloudCutoutError("no Photoroom API key set")
        url = "https://sdk.photoroom.com/v1/segment"
        headers["x-api-key"] = key
        files = {"image_file": ("garment.jpg", jpeg, "image/jpeg")}
    elif provider == "rembg_server":
        base = str(cfg.get("server_url", "")).strip().rstrip("/")
        if not base.startswith(("http://", "https://")):
            raise CloudCutoutError("server URL must start with http:// or https://")
        url = base + "/api/remove"
        files = {"file": ("garment.jpg", jpeg, "image/jpeg")}
    else:
        raise CloudCutoutError(f"unknown provider '{provider}'")

    body, content_type = _encode_multipart(fields, files)
    headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    context = _ssl_context() if url.startswith("https://") else None
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(300).decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001
            pass
        hints = {400: "request rejected", 401: "invalid API key", 402: "out of credits",
                 403: "invalid API key", 429: "rate limited"}
        message = f"HTTP {exc.code} {hints.get(exc.code, '')}".strip()
        raise CloudCutoutError(message + (f" - {detail[:120]}" if detail else "")) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CloudCutoutError(f"network error: {getattr(exc, 'reason', exc)}") from exc

    if not data.startswith(b"\x89PNG"):
        raise CloudCutoutError("the service did not return a PNG image")
    return data


# ===========================================================================
# VISION ENGINE (OpenCV DNN + classic CV, fully offline)
# ===========================================================================

SEGMENTER_FILE = "u2netp.onnx"
CLASSIFIER_FILE = "clothing_classifier.onnx"
CLASSIFIER_CONFIG_FILE = "clothing_classifier.json"

U2NET_INPUT = 320
MAX_WORK_SIDE = 1600       # long side of the working image
CUTOUT_MAX_SIDE = 1024     # long side of the saved cut-out
THUMB_SIZE = 256           # square thumbnail canvas
JPEG_QUALITY = 92
CLASSIC_WORK_SIDE = 384    # analysis size for the non-AI fallback

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CLASSIFIER_CONFIG: Dict[str, Any] = {
    "name": "MobileNetV3 (ImageNet)",
    "input_size": 224,
    "mean": list(IMAGENET_MEAN),
    "std": list(IMAGENET_STD),
    # "imagenet" -> 1000-way output mapped through IMAGENET_CATEGORY_MAP.
    # A list of category names -> a custom fine-tuned head in that order.
    "labels": "imagenet",
}

SEGMENTER_LABELS: Dict[str, str] = {
    "u2netp": "On-device AI",
    "colorkey": "On-device · colour key",
    "grabcut": "On-device · GrabCut",
    "none": "Original photo",
}


def segmenter_label(method: str) -> str:
    if method.startswith("cloud:"):
        return CLOUD_PROVIDERS.get(method.split(":", 1)[1], "Cloud")
    return SEGMENTER_LABELS.get(method, method)


class VisionError(Exception):
    """User-facing vision pipeline failure (bad image, encode failure...)."""


@dataclass
class ScanResult:
    item_id: str
    original_rel: str
    cutout_rel: str
    thumb_rel: str
    colors: List[Dict[str, Any]]
    primary_hex: Optional[str]
    color_family: str
    pattern: str
    ranking: List[Tuple[str, float]]
    confidence: float
    segmenter: str
    classifier: str
    mask_coverage: float
    processing_ms: int
    image_size: Tuple[int, int]
    shape: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def category(self) -> str:
        return self.ranking[0][0]

    @property
    def primary_color_name(self) -> Optional[str]:
        return self.colors[0]["name"] if self.colors else None

    def score_for(self, category: str) -> float:
        for name, score in self.ranking:
            if name == category:
                return score
        return 0.0


# --- Image I/O ---------------------------------------------------------------

def load_image_bgr(path: str, exif_orientation: int = 1) -> "np.ndarray":
    """Decode an image file to an 8-bit BGR array.

    * Uses np.fromfile + cv2.imdecode so non-ASCII Windows paths work.
    * JPEGs are decoded with IMREAD_COLOR, which honours EXIF rotation.
    * PNG/WebP with transparency are cropped to their visible area and
      composited onto white (product photos, Kivy camera captures).
    * ``exif_orientation`` applies an extra rotation for sources that lost
      their EXIF data (Android gallery picks are re-encoded).
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise VisionError(f"Could not read the image file: {exc}") from exc
    if data.size == 0:
        raise VisionError("The image file is empty.")

    ext = os.path.splitext(path)[1].lower()
    flag = cv2.IMREAD_UNCHANGED if ext in (".png", ".webp") else cv2.IMREAD_COLOR
    img = cv2.imdecode(data, flag)
    if img is None:
        raise VisionError("Unsupported or corrupt image file.")

    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        alpha = img[:, :, 3]
        if int(alpha.min()) < 255:
            ys, xs = np.where(alpha > 0)
            if ys.size == 0:
                raise VisionError("The image is completely transparent.")
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            a = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            img = (rgb * a + 255.0 * (1.0 - a)).clip(0, 255).astype(np.uint8)
        else:
            img = img[:, :, :3]

    rotations = {
        3: cv2.ROTATE_180,
        6: cv2.ROTATE_90_CLOCKWISE,
        8: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if exif_orientation in rotations:
        img = cv2.rotate(img, rotations[exif_orientation])

    if min(img.shape[:2]) < 32:
        raise VisionError("The image is too small to analyse.")
    return np.ascontiguousarray(img)


def limit_size(img: "np.ndarray", max_side: int) -> "np.ndarray":
    h, w = img.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def encode_image(ext: str, img: "np.ndarray", params: Optional[List[int]] = None) -> bytes:
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise VisionError(f"Could not encode image as {ext}.")
    return buf.tobytes()


# --- Mask helpers ---------------------------------------------------------------

def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def keep_main_components(binary: "np.ndarray", min_ratio: float = 0.15) -> "np.ndarray":
    """Keep the largest blob plus any blob at least ``min_ratio`` of its size
    (a pair of shoes, or trouser legs separated by a fold)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return binary.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    keep = [i + 1 for i, area in enumerate(areas) if area >= min_ratio * largest]
    return np.isin(labels, keep).astype(np.uint8)


def fill_small_holes(binary: "np.ndarray", max_area_ratio: float = 0.03) -> "np.ndarray":
    """Fill enclosed background pockets (e.g. a white logo on a white
    backdrop). Pockets touching the frame edge - the gap between trouser legs
    or between a sleeve and the torso - are open, so they are never filled."""
    inv = (binary == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if count <= 1:
        return binary
    border_labels = set(np.unique(np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))).tolist())
    limit = max_area_ratio * binary.size
    holes = [i for i in range(1, count)
             if i not in border_labels and stats[i, cv2.CC_STAT_AREA] <= limit]
    if not holes:
        return binary
    out = binary.copy()
    out[np.isin(labels, holes)] = 1
    return out


def refine_alpha(prob: "np.ndarray") -> "np.ndarray":
    """Turn a 0..1 foreground probability map into a clean, feathered alpha."""
    h, w = prob.shape[:2]
    k = _odd(round(min(h, w) * 0.006))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = (prob > 0.5).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = keep_main_components(binary)
    binary = fill_small_holes(binary)
    alpha = cv2.GaussianBlur(binary.astype(np.float32), (k, k), 0)
    return np.clip(alpha, 0.0, 1.0)


def alpha_bbox(alpha: "np.ndarray", threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1) exclusive bounds of the foreground."""
    ys, xs = np.where(alpha > threshold)
    if ys.size == 0:
        h, w = alpha.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bgr_to_lab(bgr: "np.ndarray") -> "np.ndarray":
    """8-bit BGR -> float CIELAB (L 0..100)."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _border_samples(lab: "np.ndarray", frac: float = 0.04) -> "np.ndarray":
    h, w = lab.shape[:2]
    bh, bw = max(2, int(h * frac)), max(2, int(w * frac))
    return np.concatenate((
        lab[:bh].reshape(-1, 3), lab[-bh:].reshape(-1, 3),
        lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3),
    ))


def classic_probability(bgr: "np.ndarray") -> Tuple["np.ndarray", str]:
    """Non-AI background removal. Returns (full-size 0..1 map, method).

    1. Estimate the backdrop colour from the image border (CIELAB median).
    2. Colour-key: Otsu-threshold each pixel's distance from that colour.
    3. If the border is uniform and the key separates cleanly ("colorkey"),
       recompute the key at full resolution - milliseconds, crisp edges.
       Otherwise refine with GrabCut seeded by the key mask ("grabcut"),
       on a small copy with few iterations to stay fast on phones.
    """
    h, w = bgr.shape[:2]
    scale = min(1.0, CLASSIC_WORK_SIDE / float(max(h, w)))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr.copy()
    sh, sw = small.shape[:2]

    lab = cv2.GaussianBlur(_bgr_to_lab(small), (5, 5), 0)
    border = _border_samples(lab)
    backdrop = np.median(border, axis=0)
    # 90th percentile, not median: lighting gradients and sensor noise raise
    # every border pixel a little (median ~6 on plain and busy backdrops
    # alike), while clutter puts a minority of border pixels far away.
    border_spread = float(np.percentile(np.linalg.norm(border - backdrop, axis=1), 90))
    dist = np.linalg.norm(lab - backdrop, axis=2)
    p99 = max(float(np.percentile(dist, 99)), 1e-6)
    dist8 = np.clip(dist * (255.0 / p99), 0, 255).astype(np.uint8)
    otsu, guess = cv2.threshold(dist8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = float(otsu) / 255.0 * p99
    guess = keep_main_components(guess)
    coverage = float(guess.mean())
    plausible = 0.03 <= coverage <= 0.97

    if plausible and border_spread < 15.0 and threshold > 2.0 * max(border_spread, 1.0):
        full_lab = cv2.GaussianBlur(_bgr_to_lab(bgr), (3, 3), 0)
        fg = (np.linalg.norm(full_lab - backdrop, axis=2) > threshold).astype(np.uint8)
        return keep_main_components(fg).astype(np.float32), "colorkey"

    mask = np.zeros((sh, sw), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    if plausible:
        mask[:] = np.where(guess > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD)
        cv2.grabCut(small, mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    else:
        mx, my = max(1, int(sw * 0.04)), max(1, int(sh * 0.04))
        rect = (mx, my, max(1, sw - 2 * mx), max(1, sh - 2 * my))
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    fg = keep_main_components(fg)
    return cv2.resize(fg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR), "grabcut"


# --- Colour analysis ------------------------------------------------------------

_named_lab_cache: Optional["np.ndarray"] = None


def hex_to_rgb01(hex_code: str) -> Tuple[float, float, float]:
    h = hex_code.lstrip("#")
    return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0


def rgb01_to_hex(rgb: "np.ndarray") -> str:
    r, g, b = (int(round(float(c) * 255)) for c in np.clip(rgb, 0, 1))
    return f"#{r:02X}{g:02X}{b:02X}"


def rgb01_to_lab(rgb: "np.ndarray") -> "np.ndarray":
    """(N,3) RGB in 0..1 -> (N,3) CIELAB (L 0..100, a/b approx -128..127)."""
    arr = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def _named_lab() -> "np.ndarray":
    global _named_lab_cache
    if _named_lab_cache is None:
        rgb = np.array([hex_to_rgb01(hx) for _name, hx in NAMED_COLORS], dtype=np.float32)
        _named_lab_cache = rgb01_to_lab(rgb)
    return _named_lab_cache


def nearest_color_name(lab: "np.ndarray") -> str:
    distances = np.linalg.norm(_named_lab() - np.asarray(lab, dtype=np.float32)[None, :], axis=1)
    return NAMED_COLORS[int(np.argmin(distances))][0]


def _merge_distance(lab_a: "np.ndarray", lab_b: "np.ndarray") -> float:
    """Lightness-discounted ΔE so a shadowed fold merges with the lit fabric."""
    d = np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)
    return float(math.sqrt((0.5 * d[0]) ** 2 + d[1] ** 2 + d[2] ** 2))


def extract_colors(bgr: "np.ndarray", alpha: "np.ndarray",
                   k: int = 5, max_colors: int = 3) -> List[Dict[str, Any]]:
    """Dominant garment colours by pixel frequency.

    K-Means in CIELAB on opaque garment pixels only; clusters that differ
    mainly in lightness (shadows/folds) are merged; each result carries its
    share of the garment, hex code and a consumer-friendly name.
    """
    pixels = bgr[alpha > 0.9]
    if pixels.shape[0] < 50:
        pixels = bgr.reshape(-1, 3)
    rng = np.random.default_rng(0)
    if pixels.shape[0] > 12000:
        pixels = pixels[rng.choice(pixels.shape[0], 12000, replace=False)]

    rgb = pixels[:, ::-1].astype(np.float32) / 255.0
    lab = rgb01_to_lab(rgb).astype(np.float32)
    k_eff = int(min(k, max(1, lab.shape[0] // 10)))

    cv2.setRNGSeed(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _compactness, labels, _centers = cv2.kmeans(
        lab, k_eff, None, criteria, 2, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()

    clusters: List[Dict[str, Any]] = []
    for i in range(k_eff):
        member = labels == i
        count = int(member.sum())
        if count == 0:
            continue
        clusters.append({
            "lab": lab[member].mean(axis=0).astype(np.float64),
            "rgb": rgb[member].mean(axis=0).astype(np.float64),
            "count": count,
        })
    clusters.sort(key=lambda c: c["count"], reverse=True)

    merged: List[Dict[str, Any]] = []
    for cluster in clusters:
        target = next((m for m in merged if _merge_distance(m["lab"], cluster["lab"]) < 14.0), None)
        if target is None:
            merged.append(dict(cluster))
            continue
        total = target["count"] + cluster["count"]
        for key in ("lab", "rgb"):
            target[key] = (target[key] * target["count"] + cluster[key] * cluster["count"]) / total
        target["count"] = total
    merged.sort(key=lambda c: c["count"], reverse=True)

    grand_total = float(sum(c["count"] for c in merged)) or 1.0
    result: List[Dict[str, Any]] = []
    for cluster in merged:
        share = cluster["count"] / grand_total
        if share < 0.05 and result:
            continue
        r, g, b = (float(v) for v in np.clip(cluster["rgb"], 0, 1))
        hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
        L, A, B = (float(v) for v in cluster["lab"])
        result.append({
            "hex": rgb01_to_hex(cluster["rgb"]),
            "name": nearest_color_name(cluster["lab"]),
            "hsv": {"h": round(hue * 360.0, 1), "s": round(sat, 3), "v": round(val, 3)},
            "lab": {"L": round(L, 2), "a": round(A, 2), "b": round(B, 2)},
            "share": round(share, 3),
        })
        if len(result) >= max_colors:
            break
    return result


def color_family(color: Optional[Dict[str, Any]]) -> str:
    if not color:
        return "neutral"
    chroma = math.hypot(color["lab"]["a"], color["lab"]["b"])
    if color["name"] in NEUTRAL_COLOR_NAMES or chroma < 10.0:
        return "neutral"
    if color["name"] in EARTH_COLOR_NAMES:
        return "earth"
    hue = color["hsv"]["h"]
    return "warm" if hue < 75.0 or hue >= 330.0 else "cool"


def is_denim_like(color: Optional[Dict[str, Any]]) -> bool:
    if not color:
        return False
    hsv = color["hsv"]
    return 195.0 <= hsv["h"] <= 235.0 and 0.2 <= hsv["s"] <= 0.8 and 0.15 <= hsv["v"] <= 0.8


def detect_pattern(bgr: "np.ndarray", alpha: "np.ndarray", colors: List[Dict[str, Any]]) -> str:
    """'solid' / 'printed' / 'unknown' from interior edge density.

    The mask is eroded first so the garment's own outline is not counted.
    """
    work = limit_size(bgr, 512)
    a = cv2.resize(alpha, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_LINEAR)
    inner = (a > 0.9).astype(np.uint8)
    k = _odd(round(min(work.shape[:2]) * 0.04))
    inner = cv2.erode(inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if int(inner.sum()) < 500:
        return "unknown"
    gray = cv2.GaussianBlur(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    density = float((edges[inner > 0] > 0).mean())
    top_share = float(colors[0]["share"]) if colors else 0.0
    if density > 0.09 or (top_share < 0.45 and density > 0.04):
        return "printed"
    return "solid"


# --- Contour / pixel-ratio shape analyser ------------------------------------------

def silhouette_features(alpha: "np.ndarray") -> Optional[Dict[str, Any]]:
    """Shape descriptors of a flat-laid garment from its mask.

    Pixel ratios: aspect, fill and band widths (shoulders / waist / hem).
    Run analysis: how many separate runs each row splits into (sleeves,
    legs) and where the gaps sit. Contour metrics: solidity (area / convex
    hull area) and extent (area / bounding-box area) of the main outline.
    """
    mask = (alpha > 0.5).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if ys.size < 100:
        return None
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    n = 128  # grid resolution: fine enough to keep a narrow V between legs
    grid = cv2.resize(crop.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA) > 0.5
    rows = grid.mean(axis=1)

    def band(a: float, b: float) -> float:
        lo, hi = int(a * n), max(int(b * n), int(a * n) + 1)
        return float(rows[lo:hi].mean())

    def row_runs(row: "np.ndarray") -> Tuple["np.ndarray", "np.ndarray"]:
        diff = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        return np.where(diff == 1)[0], np.where(diff == -1)[0]

    upper, middle, lower = band(0.12, 0.35), band(0.40, 0.60), band(0.70, 0.92)

    # Leg gap: in the bottom 30%, most rows split into exactly two runs with
    # the gap near the middle, while the top (waistband) is one solid run.
    # Sleeve gaps sit off-centre, so long-sleeve tops do not trigger this.
    lower_rows = grid[int(0.70 * n):int(0.97 * n)]
    centred_gaps = 0
    for row in lower_rows:
        starts, ends = row_runs(row)
        if len(starts) == 2:
            gap_mid = (ends[0] + starts[1]) / 2.0 / n
            if 0.35 <= gap_mid <= 0.65:
                centred_gaps += 1
    top_runs = [len(row_runs(row)[0]) for row in grid[int(0.02 * n):int(0.15 * n)]]
    solid_top = bool(top_runs) and float(np.median(top_runs)) == 1.0
    leg_gap = solid_top and centred_gaps >= 0.6 * max(1, len(lower_rows))

    runs = [len(row_runs(row)[0]) for row in grid[int(0.45 * n):int(0.85 * n)]]

    solidity, extent = 1.0, float(crop.mean())
    contours, _hier = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(main))
        hull_area = float(cv2.contourArea(cv2.convexHull(main)))
        if hull_area > 0:
            solidity = area / hull_area
        extent = area / float(h * w)

    return {
        "aspect": h / float(w),
        "fill": float(crop.mean()),
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "leg_gap": leg_gap,
        "median_runs": float(np.median(runs)) if runs else 1.0,
        "sleeve_ratio": upper / max(lower, 1e-3),
        "solidity": round(solidity, 4),
        "extent": round(extent, 4),
    }


def heuristic_scores(features: Optional[Dict[str, Any]],
                     primary: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Normalised category distribution from silhouette rules."""
    s = {c: 0.05 for c in CATEGORIES}
    if features is None:
        for c in ("T-Shirt", "Shirt", "Sweater", "Hoodie", "Jacket", "Pants"):
            s[c] += 1.0
    else:
        a = features["aspect"]
        # Bottoms are identified only by a gap in the CENTRE of the lower
        # half (between the legs). Gaps at the sides are sleeves.
        if features["leg_gap"]:
            if a < 1.05:
                s["Shorts"] += 3.0
            else:
                s["Pants"] += 2.0
                s["Trouser"] += 1.2
                s["Jeans"] += 1.2
            if is_denim_like(primary):
                s["Jeans"] += 2.5
        elif features["median_runs"] >= 2:
            # Long sleeves lying beside the torso: 2-3 separate runs per row
            # (one sleeve may touch the body).
            for c, v in (("Hoodie", 1.2), ("Sweater", 1.2), ("Jacket", 1.2), ("Shirt", 1.0)):
                s[c] += v
            if a > 1.45:
                s["Coat"] += 1.8
        elif a >= 1.7 and features["lower"] <= features["upper"] * 1.1:
            # Tall, non-flaring rectangle: trousers lying with legs together.
            s["Pants"] += 1.2
            s["Trouser"] += 0.8
            s["Jeans"] += 0.8 + (1.5 if is_denim_like(primary) else 0.0)
            s["Coat"] += 0.8
            s["Dress"] += 0.4
        elif features["sleeve_ratio"] > 1.25 and a < 1.25:
            # Wide shoulder line with short sleeves sticking out.
            s["T-Shirt"] += 2.2
            s["Shirt"] += 1.0
            s["Sweater"] += 0.4
        elif features["lower"] > features["upper"] * 1.2 and 0.6 <= a <= 1.5:
            # Trapezoid flaring towards the hem.
            s["Skirt"] += 2.0
            s["Dress"] += 0.6
        elif a > 1.6 and features["lower"] >= features["middle"]:
            s["Dress"] += 2.0
            s["Coat"] += 1.0
        elif a < 0.7 and features["fill"] > 0.5:
            s["Shoes"] += 1.5
            s["Accessory"] += 0.5
        else:
            for c, v in (("T-Shirt", 1.0), ("Shirt", 1.0), ("Sweater", 0.8),
                         ("Hoodie", 0.8), ("Jacket", 0.8), ("Pants", 0.5)):
                s[c] += v
    total = sum(s.values())
    return {c: v / total for c, v in s.items()}


def softmax(x: "np.ndarray") -> "np.ndarray":
    x = np.asarray(x, dtype=np.float64)
    if np.all(x >= 0) and abs(float(x.sum()) - 1.0) < 1e-3:
        return x  # already probabilities
    e = np.exp(x - x.max())
    return e / e.sum()


class VisionEngine:
    """Garment analysis: cut-out, colours, pattern, category.

    Models are loaded lazily on the first scan (inside the worker thread) and
    cached. A single lock guards the DNN nets because cv2.dnn.Net instances
    are not thread-safe.
    """

    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self.seg_path = os.path.join(models_dir, SEGMENTER_FILE)
        self.cls_path = os.path.join(models_dir, CLASSIFIER_FILE)
        self.cls_cfg_path = os.path.join(models_dir, CLASSIFIER_CONFIG_FILE)
        self._net_lock = threading.Lock()
        self._seg_net = None
        self._seg_state = "unloaded"   # unloaded | ready | missing | error
        self._cls_net = None
        self._cls_state = "unloaded"
        self._cls_cfg: Dict[str, Any] = dict(DEFAULT_CLASSIFIER_CONFIG)

    # -- status -----------------------------------------------------------
    def status_text(self) -> str:
        if self._seg_state == "error":
            seg = "Classic cut-out (u2netp.onnx failed to load - see log)"
        elif os.path.isfile(self.seg_path):
            seg = "AI cut-out (U2-Net-p)"
        else:
            seg = "Classic cut-out (colour key / GrabCut)"
        if self._cls_state == "error":
            cls = "shape analyser (classifier failed to load)"
        elif os.path.isfile(self.cls_path):
            cls = f"{self._read_cls_config().get('name', 'Classifier')} + shape analyser"
        else:
            cls = "shape analyser"
        return f"{seg} · {cls}"

    # -- model loading ----------------------------------------------------
    def _read_cls_config(self) -> Dict[str, Any]:
        cfg = dict(DEFAULT_CLASSIFIER_CONFIG)
        loaded = read_json(self.cls_cfg_path, None) if os.path.isfile(self.cls_cfg_path) else None
        if isinstance(loaded, dict):
            cfg.update({k: v for k, v in loaded.items() if k in DEFAULT_CLASSIFIER_CONFIG})
        return cfg

    def _load_segmenter(self):
        if self._seg_state == "unloaded":
            if not os.path.isfile(self.seg_path):
                self._seg_state = "missing"
            else:
                try:
                    self._seg_net = cv2.dnn.readNetFromONNX(self.seg_path)
                    self._seg_state = "ready"
                    Logger.info(f"{LOG_TAG}: loaded segmenter {self.seg_path}")
                except cv2.error as exc:
                    self._seg_state = "error"
                    Logger.error(f"{LOG_TAG}: could not load {self.seg_path}: {exc}")
        return self._seg_net if self._seg_state == "ready" else None

    def _load_classifier(self):
        if self._cls_state == "unloaded":
            if not os.path.isfile(self.cls_path):
                self._cls_state = "missing"
            else:
                try:
                    self._cls_cfg = self._read_cls_config()
                    self._cls_net = cv2.dnn.readNetFromONNX(self.cls_path)
                    self._cls_state = "ready"
                    Logger.info(f"{LOG_TAG}: loaded classifier {self.cls_path}")
                except cv2.error as exc:
                    self._cls_state = "error"
                    Logger.error(f"{LOG_TAG}: could not load {self.cls_path}: {exc}")
        return self._cls_net if self._cls_state == "ready" else None

    # -- segmentation -----------------------------------------------------
    def _u2net_probability(self, bgr: "np.ndarray") -> Optional["np.ndarray"]:
        net = self._load_segmenter()
        if net is None:
            return None
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (U2NET_INPUT, U2NET_INPUT), interpolation=cv2.INTER_AREA).astype(np.float32)
        x /= max(float(x.max()), 1e-6)
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        blob = np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...], dtype=np.float32)

        with self._net_lock:
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())
        if not isinstance(outs, (list, tuple)):
            outs = [outs]

        # U2-Net exposes the fused map d0 plus six side outputs, and OpenCV
        # does not guarantee their order. The fused map is the most decisive
        # one (values pushed furthest from 0.5), so pick by that.
        best, best_score = None, -1.0
        for out in outs:
            arr = np.asarray(out, dtype=np.float32)
            if arr.size != U2NET_INPUT * U2NET_INPUT:
                continue
            arr = arr.reshape(U2NET_INPUT, U2NET_INPUT)
            score = float(np.mean(np.abs(arr - 0.5)))
            if score > best_score:
                best, best_score = arr, score
        if best is None:
            raise VisionError("U2-Net returned an unexpected output shape.")
        lo, hi = float(best.min()), float(best.max())
        if hi - lo < 1e-6:
            return None
        pred = (best - lo) / (hi - lo)
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

    def _cloud_alpha(self, bgr: "np.ndarray", cfg: Dict[str, Any]) -> "np.ndarray":
        """Alpha from a cloud provider, aligned to ``bgr``. Raises on failure."""
        provider = cfg.get("provider", "local")
        jpeg = encode_image(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        png = cloud_cutout(provider, jpeg, cfg)
        rgba = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise CloudCutoutError("the cut-out has no transparency")
        h, w = bgr.shape[:2]
        rh, rw = rgba.shape[:2]
        if abs((rw / float(rh)) - (w / float(h))) > 0.02 * (w / float(h)):
            raise CloudCutoutError("the cut-out was cropped and cannot be aligned")
        alpha = rgba[:, :, 3].astype(np.float32) / 255.0
        if (rh, rw) != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return alpha

    def segment(self, bgr: "np.ndarray",
                cutout_cfg: Optional[Dict[str, Any]] = None) -> Tuple["np.ndarray", str, List[str]]:
        """Return (alpha 0..1 float32, method, notes). Never raises for
        model/network problems - it degrades cloud (if enabled) -> U2-Net ->
        colour key / GrabCut -> full frame."""
        notes: List[str] = []
        cfg = cutout_cfg or {}
        provider = cfg.get("provider", "local")

        if provider != "local" and provider in CLOUD_PROVIDERS:
            label = CLOUD_PROVIDERS[provider]
            if not cloud_provider_ready(cfg):
                notes.append(f"{label} is selected but not configured; used on-device.")
            else:
                started = time.perf_counter()
                try:
                    alpha = self._cloud_alpha(bgr, cfg)
                    coverage = float(alpha.mean())
                    if 0.03 <= coverage <= 0.97:
                        ms = int((time.perf_counter() - started) * 1000)
                        notes.append(f"{label} round-trip {ms} ms.")
                        return alpha, f"cloud:{provider}", notes
                    notes.append(f"{label} mask rejected (coverage {coverage:.0%}); used on-device.")
                except (CloudCutoutError, VisionError) as exc:
                    Logger.warning(f"{LOG_TAG}: cloud cut-out failed: {exc}")
                    notes.append(f"{label} unavailable ({exc}); used on-device.")

        try:
            prob = self._u2net_probability(bgr)
        except (cv2.error, VisionError) as exc:
            Logger.error(f"{LOG_TAG}: U2-Net inference failed, using classic cut-out: {exc}")
            self._seg_state = "error"
            prob = None
        if prob is not None:
            alpha = refine_alpha(prob)
            coverage = float(alpha.mean())
            if 0.03 <= coverage <= 0.97:
                return alpha, "u2netp", notes
            notes.append(f"AI mask rejected (coverage {coverage:.0%}); used the classic cut-out.")

        try:
            prob, method = classic_probability(bgr)
            alpha = refine_alpha(prob)
            coverage = float(alpha.mean())
            if 0.03 <= coverage <= 0.97:
                return alpha, method, notes
            notes.append(f"Classic mask rejected (coverage {coverage:.0%}).")
        except cv2.error as exc:
            Logger.error(f"{LOG_TAG}: classic segmentation failed: {exc}")
            notes.append("Classic cut-out failed.")

        notes.append("Background could not be removed; kept the full photo.")
        return np.ones(bgr.shape[:2], np.float32), "none", notes

    # -- classification ---------------------------------------------------
    def _cnn_scores(self, bgr: "np.ndarray", alpha: "np.ndarray") -> Tuple[Optional[Dict[str, float]], float, bool]:
        """Return (category distribution, clothing mass, is_custom_head)."""
        net = self._load_classifier()
        if net is None:
            return None, 0.0, False
        cfg = self._cls_cfg
        size = int(cfg.get("input_size", 224))

        # Garment on white, square-padded - matches product-shot statistics.
        a = alpha[:, :, None]
        comp = (bgr.astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)
        x0, y0, x1, y1 = alpha_bbox(alpha)
        crop = comp[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        canvas = np.full((side, side, 3), 255, np.uint8)
        oy, ox = (side - ch) // 2, (side - cw) // 2
        canvas[oy:oy + ch, ox:ox + cw] = crop

        rgb = cv2.cvtColor(cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array(cfg["mean"], np.float32)) / np.array(cfg["std"], np.float32)
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32)

        with self._net_lock:
            net.setInput(blob)
            out = net.forward()
        probs = softmax(np.asarray(out).reshape(-1))

        labels = cfg.get("labels", "imagenet")
        dist = {c: 0.0 for c in CATEGORIES}
        if isinstance(labels, list):
            if len(labels) != probs.size:
                raise VisionError("Classifier label count does not match its output size.")
            for name, p in zip(labels, probs):
                if name in dist:
                    dist[name] += float(p)
            mass, custom = float(sum(dist.values())), True
        else:
            if probs.size != 1000:
                raise VisionError(f"Expected 1000 ImageNet outputs, got {probs.size}.")
            for idx, weights in IMAGENET_CATEGORY_MAP.items():
                for name, wgt in weights.items():
                    dist[name] += float(probs[idx]) * wgt
            mass, custom = float(sum(dist.values())), False
        if mass <= 1e-9:
            return None, 0.0, custom
        return {c: v / mass for c, v in dist.items()}, mass, custom

    def classify(self, bgr: "np.ndarray", alpha: "np.ndarray",
                 primary: Optional[Dict[str, Any]]) -> Tuple[List[Tuple[str, float]], str, Dict[str, Any]]:
        features = silhouette_features(alpha)
        heur = heuristic_scores(features, primary)
        final, method = dict(heur), "shape-analyser"
        try:
            cnn, mass, custom = self._cnn_scores(bgr, alpha)
        except (cv2.error, VisionError) as exc:
            Logger.error(f"{LOG_TAG}: classifier inference failed: {exc}")
            self._cls_state = "error"
            cnn, mass, custom = None, 0.0, False
        if cnn is not None:
            # Trust a fine-tuned head heavily; trust ImageNet in proportion to
            # how much probability it put on clothing classes at all.
            weight = 0.85 if custom else min(0.7, mass * 1.5)
            final = {c: weight * cnn.get(c, 0.0) + (1.0 - weight) * heur.get(c, 0.0) for c in CATEGORIES}
            method = f"{self._cls_cfg.get('name', 'classifier')} + shape-analyser"
        ranking = sorted(((c, round(float(v), 4)) for c, v in final.items()),
                         key=lambda kv: kv[1], reverse=True)
        shape = {}
        if features is not None:
            shape = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in features.items()}
        return ranking, method, shape

    # -- full pipeline ----------------------------------------------------
    def process_file(
        self,
        src_path: str,
        storage: StorageManager,
        progress: Callable[[str], None],
        exif_orientation: int = 1,
        cutout_cfg: Optional[Dict[str, Any]] = None,
    ) -> ScanResult:
        started = time.perf_counter()

        progress("Reading photo")
        bgr = limit_size(load_image_bgr(src_path, exif_orientation), MAX_WORK_SIDE)
        h, w = bgr.shape[:2]

        cfg = cutout_cfg or {}
        if cfg.get("provider", "local") != "local" and cloud_provider_ready(cfg):
            progress(f"Removing background via {CLOUD_PROVIDERS.get(cfg['provider'], 'cloud')}")
        else:
            progress("Removing background")
        alpha, seg_method, notes = self.segment(bgr, cfg)

        progress("Reading colours")
        colors = extract_colors(bgr, alpha)
        primary = colors[0] if colors else None
        family = color_family(primary)
        pattern = detect_pattern(bgr, alpha, colors)

        progress("Identifying garment")
        ranking, cls_method, shape = self.classify(bgr, alpha, primary)

        progress("Polishing cut-out")
        item_id = uuid.uuid4().hex
        original_rel = f"{IMG_ORIGINAL_DIR}/{item_id}_original.jpg"
        cutout_rel = f"{IMG_CUTOUT_DIR}/{item_id}_cutout.png"
        thumb_rel = f"{IMG_THUMB_DIR}/{item_id}_thumb.png"

        atomic_write_bytes(storage.path(original_rel),
                           encode_image(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]))

        x0, y0, x1, y1 = alpha_bbox(alpha, 0.05)
        pad = int(0.04 * max(x1 - x0, y1 - y0))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = (alpha * 255.0).clip(0, 255).astype(np.uint8)
        cutout = limit_size(bgra[y0:y1, x0:x1], CUTOUT_MAX_SIDE)
        atomic_write_bytes(storage.path(cutout_rel), encode_image(".png", cutout))

        th, tw = cutout.shape[:2]
        scale = THUMB_SIZE / float(max(th, tw))
        small = cv2.resize(cutout, (max(1, int(tw * scale)), max(1, int(th * scale))),
                           interpolation=cv2.INTER_AREA)
        thumb = np.zeros((THUMB_SIZE, THUMB_SIZE, 4), np.uint8)
        oy, ox = (THUMB_SIZE - small.shape[0]) // 2, (THUMB_SIZE - small.shape[1]) // 2
        thumb[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
        atomic_write_bytes(storage.path(thumb_rel), encode_image(".png", thumb))

        return ScanResult(
            item_id=item_id,
            original_rel=original_rel,
            cutout_rel=cutout_rel,
            thumb_rel=thumb_rel,
            colors=colors,
            primary_hex=primary["hex"] if primary else None,
            color_family=family,
            pattern=pattern,
            ranking=ranking,
            confidence=ranking[0][1],
            segmenter=seg_method,
            classifier=cls_method,
            mask_coverage=round(float(alpha.mean()), 4),
            processing_ms=int((time.perf_counter() - started) * 1000),
            image_size=(w, h),
            shape=shape,
            notes=notes,
        )


def build_item_record(
    result: ScanResult,
    category: str,
    user_confirmed: bool,
    price: Optional[float],
    currency: str,
) -> Dict[str, Any]:
    """Assemble a wardrobe/items.json entry (schema v1)."""
    meta = CATEGORY_META[category]
    now = utc_now_iso()
    color_name = result.primary_color_name
    return {
        "item_id": result.item_id,
        "created_at": now,
        "updated_at": now,
        "display_name": f"{color_name} {category}" if color_name else category,
        "images": {
            "original": result.original_rel,
            "cutout": result.cutout_rel,
            "thumb": result.thumb_rel,
        },
        "category": category,
        "category_confidence": round(result.score_for(category), 3),
        "user_confirmed": user_confirmed,
        "slot": meta["slot"],
        "layer_level": meta["layer_level"],
        "can_layer_over": layers_under(category),
        "colors": result.colors,
        "primary_hex": result.primary_hex,
        "color_family": result.color_family,
        "pattern": result.pattern,
        "warmth_index": meta["warmth"],
        "fabric_tags": guess_fabric_tags(category, result.pattern),
        "formality": meta["formality"],
        "occasions": list(meta["occasions"]),
        "seasons": list(meta["seasons"]),
        "purchase_price": price,
        "currency": currency,
        "purchase_date": None,
        "brand": None,
        "wear_count": 0,
        "last_worn": None,
        # With zero wears, cost-per-wear equals the purchase price.
        "cost_per_wear": price,
        "status": "active",
        "favorite": False,
        "notes": "",
        "vision": {
            "segmenter": result.segmenter,
            "classifier": result.classifier,
            "top_guess": result.category,
            "top_guess_score": result.confidence,
            "mask_coverage": result.mask_coverage,
            "processing_ms": result.processing_ms,
            "source_size": list(result.image_size),
            "shape": result.shape,
        },
    }


# ===========================================================================
# BIOMETRIC PROFILE ENGINE (face, silhouette ratios, skin tone)
# Always on-device: body photos are never sent to a cloud provider.
# ===========================================================================

#: Optional OpenCV Zoo YuNet face detector (more robust than Haar). Used when
#: present in assets/models/; otherwise Haar cascades, then a skin-blob search.
FACE_YUNET_FILE = "face_detection_yunet_2023mar.onnx"
PROFILE_WORK_SIDE = 1280

#: Where body landmarks are searched, in head-heights (H) below the chin, for
#: the classic ~7.5-head adult figure. Bands, not points, because real
#: proportions vary: (start, end, pick widest|narrowest row).
BODY_BANDS: Dict[str, Tuple[float, float, str]] = {
    "shoulder": (0.25, 0.95, "max"),
    "waist": (1.45, 2.45, "min"),
    "hip": (2.55, 3.45, "max"),
}

#: A front silhouette's shoulder line runs across the deltoids, so it reads
#: ~8% wider than the skeletal shoulder width that body-shape systems compare
#: with the hips. Applied only when classifying the shape.
SHOULDER_SILHOUETTE_FACTOR = 1.08

BODY_POINT_KEYS: Tuple[str, ...] = ("shoulder_l", "shoulder_r", "waist_l", "waist_r", "hip_l", "hip_r")

BODY_SHAPE_LABELS: Dict[str, str] = {
    "hourglass": "Hourglass",
    "inverted_triangle": "Inverted triangle",
    "triangle": "Triangle",
    "rectangle": "Rectangle",
    "oval": "Oval",
}

BODY_SHAPE_TIPS: Dict[str, str] = {
    "hourglass": "Balanced shoulders and hips with a defined waist.",
    "inverted_triangle": "Shoulders broader than hips.",
    "triangle": "Hips broader than shoulders.",
    "rectangle": "Shoulders, waist and hips fairly aligned.",
    "oval": "Fullest through the middle.",
}

#: Skin-tone thresholds. ITA (Individual Typology Angle) bands follow the
#: dermatology convention; hue angle h = atan2(b*, a*) separates pinker
#: (cool) from more golden (warm) skin. Measured facial skin clusters around
#: h = 55-62 deg, so the neutral band is centred there.
ITA_DEPTH_BANDS: Tuple[Tuple[float, str], ...] = ((41.0, "light"), (28.0, "medium"), (10.0, "tan"))
UNDERTONE_COOL_MAX_HUE = 52.0
UNDERTONE_WARM_MIN_HUE = 61.0

#: General colour-theory guidance per undertone (names from NAMED_COLORS).
PALETTE_RULES: Dict[str, Dict[str, List[str]]] = {
    "warm": {
        "best": ["Camel Brown", "Olive Green", "Rust Orange", "Mustard Yellow", "Terracotta",
                 "Coral", "Emerald Green", "Teal", "Antique Gold"],
        "neutrals": ["Ivory", "Cream", "Camel Brown", "Mocha Brown", "Khaki"],
        "avoid": ["Optic White", "Hot Pink", "Royal Purple", "Ash Grey"],
    },
    "cool": {
        "best": ["Navy Blue", "Royal Blue", "Emerald Green", "Burgundy", "Lavender", "Plum",
                 "Hot Pink", "Sky Blue", "Teal"],
        "neutrals": ["Optic White", "Heather Grey", "Charcoal Grey", "Navy Blue", "Jet Black"],
        "avoid": ["Mustard Yellow", "Rust Orange", "Bright Orange", "Camel Brown"],
    },
    "neutral": {
        "best": ["Dusty Rose", "Sage Green", "Teal", "Denim Blue", "Navy Blue", "Plum",
                 "Mint Green", "Coral", "Emerald Green"],
        "neutrals": ["Sand Beige", "Heather Grey", "Navy Blue", "Charcoal Grey", "Ivory"],
        "avoid": ["Bright Orange", "Sunflower Yellow"],
    },
}
DEEP_BEST: List[str] = ["Optic White", "Royal Blue", "Crimson Red", "Emerald Green", "Hot Pink", "Antique Gold"]
DEEP_AVOID: List[str] = ["Ash Grey", "Sand Beige", "Powder Blue"]

NAMED_COLOR_HEX: Dict[str, str] = dict(NAMED_COLORS)


def _clip_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x, y, bw, bh = box
    x, y = max(0, int(x)), max(0, int(y))
    bw, bh = max(1, min(int(bw), w - x)), max(1, min(int(bh), h - y))
    return x, y, bw, bh


def _skin_mask(bgr: "np.ndarray") -> "np.ndarray":
    """Classic YCrCb skin range (works across skin tones; excludes most fabric)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))


def _skin_blob_face(bgr: "np.ndarray", person_mask: Optional["np.ndarray"]) -> Optional[Tuple[int, int, int, int]]:
    """Fallback face finder: the largest skin-coloured blob in the top of the person."""
    h, w = bgr.shape[:2]
    skin = (_skin_mask(bgr) > 0).astype(np.uint8)
    if person_mask is not None and int(person_mask.sum()) > 0:
        skin &= person_mask.astype(np.uint8)
        ys = np.where(person_mask.any(axis=1))[0]
        top, bottom = int(ys.min()), int(ys.max())
        limit = top + int(0.35 * (bottom - top))
    else:
        limit = int(0.45 * h)
    skin[limit:, :] = 0
    k = _odd(round(min(h, w) * 0.008))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    if count <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = (int(v) for v in stats[idx])
    if area < 0.0015 * h * w:
        return None
    if bh > 1.45 * bw:  # blob includes the neck - keep a face-like aspect
        bh = int(1.3 * bw)
    return _clip_box((x, y, bw, bh), w, h)


_haar_lock = threading.Lock()
_haar_cache: Dict[str, Any] = {}


def _haar_cascade():
    """Load OpenCV's bundled frontal-face cascade once (parsing the XML takes
    over a second on slow devices). None when cascades are not available."""
    with _haar_lock:
        if "cascade" not in _haar_cache:
            cascade = None
            data = getattr(cv2, "data", None)
            path = os.path.join(data.haarcascades, "haarcascade_frontalface_default.xml") if data else ""
            if path and os.path.isfile(path):
                loaded = cv2.CascadeClassifier(path)
                if not loaded.empty():
                    cascade = loaded
            _haar_cache["cascade"] = cascade
        return _haar_cache["cascade"]


def detect_face(bgr: "np.ndarray", person_mask: Optional["np.ndarray"] = None,
                yunet_path: Optional[str] = None) -> Tuple[Optional[Tuple[int, int, int, int]], str]:
    """Return ((x, y, w, h) in pixels, method) for the main face, or (None, "none")."""
    h, w = bgr.shape[:2]

    if yunet_path and os.path.isfile(yunet_path) and hasattr(cv2, "FaceDetectorYN"):
        try:
            detector = cv2.FaceDetectorYN.create(yunet_path, "", (w, h), 0.7, 0.3, 5000)
            _ret, faces = detector.detect(bgr)
            if faces is not None and len(faces):
                best = max(faces, key=lambda f: float(f[2] * f[3] * f[-1]))
                return _clip_box(tuple(int(round(float(v))) for v in best[:4]), w, h), "yunet"
        except cv2.error as exc:
            Logger.warning(f"{LOG_TAG}: YuNet face detection failed: {exc}")

    cascade = _haar_cascade()
    if cascade is not None:
        head_limit = 0.6 * h
        if person_mask is not None and int(person_mask.sum()) > 0:
            rows = np.where(person_mask.any(axis=1))[0]
            head_limit = rows.min() + 0.3 * (rows.max() - rows.min())
        # Search only the head region, downscaled: ~10x faster than scanning
        # the whole photo, with identical results for a standing portrait.
        search_bottom = min(h, int(head_limit + 0.08 * h))
        region = bgr[:search_bottom]
        scale = min(1.0, 480.0 / max(1, region.shape[1]))
        small = cv2.resize(region, (max(1, int(region.shape[1] * scale)), max(1, int(region.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1.0 else region
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        min_side = max(20, int(min(h, w) * 0.04 * scale))
        found = cascade.detectMultiScale(gray, 1.1, 5, minSize=(min_side, min_side))
        faces = [tuple(int(round(v / scale)) for v in f) for f in found]
        if faces:
            skin = _skin_mask(bgr) > 0
            candidates = []
            for f in faces:
                x, y, fw, fh = (int(v) for v in f)
                cy, cxx = min(h - 1, y + fh // 2), min(w - 1, x + fw // 2)
                if cy > head_limit:
                    continue  # a "face" on the torso or legs is a false positive
                if person_mask is not None and not person_mask[cy, cxx]:
                    continue
                if float(skin[y:y + fh, x:x + fw].mean()) < 0.25:
                    continue  # real faces are largely skin-coloured
                candidates.append((x, y, fw, fh))
            if candidates:
                return _clip_box(max(candidates, key=lambda f: f[2] * f[3]), w, h), "haar"

    box = _skin_blob_face(bgr, person_mask)
    if box is not None:
        return box, "skin-blob"
    return None, "none"


def classify_body_shape(shoulder: float, waist: float, hip: Optional[float]) -> Optional[str]:
    """Five-type body-shape estimate from front-silhouette widths."""
    if not hip or not shoulder or not waist:
        return None
    s = shoulder / SHOULDER_SILHOUETTE_FACTOR
    widest = max(s, hip)
    if waist >= 0.95 * widest:
        return "oval"
    if abs(s - hip) / widest <= 0.06 and waist <= 0.78 * min(s, hip):
        return "hourglass"
    if s > hip * 1.06:
        return "inverted_triangle"
    if hip > s * 1.06:
        return "triangle"
    return "rectangle"


def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return round(a / b, 3) if a and b else None


def body_metrics(shoulder: float, waist: float, hip: Optional[float]) -> Dict[str, Any]:
    return {
        "shoulder_px": round(float(shoulder), 1),
        "waist_px": round(float(waist), 1),
        "hip_px": round(float(hip), 1) if hip else None,
        "shoulder_to_waist_ratio": _ratio(shoulder, waist),
        "waist_to_hip_ratio": _ratio(waist, hip),
        "shape": classify_body_shape(shoulder, waist, hip),
    }


def measure_body(mask: "np.ndarray", face_box: Optional[Tuple[int, int, int, int]]) -> Optional[Dict[str, Any]]:
    """Pixel-ratio measurement of shoulders, waist and hips from a person mask.

    For each row in a landmark band it takes the run of foreground pixels that
    contains the body's centre line, so arms held slightly away from the body
    are excluded. Widths are smoothed over 5 rows before picking the widest
    (shoulders, hips) or narrowest (waist) row.
    """
    ys, xs = np.where(mask > 0)
    if ys.size < 500:
        return None
    img_h, img_w = mask.shape[:2]
    top, bottom = int(ys.min()), int(ys.max())
    notes: List[str] = []

    if face_box is not None:
        fx, fy, fw, fh = face_box
        head = fh * 1.15
        chin = fy + fh
        cx = fx + fw / 2.0
    else:
        head = (bottom - top) / 7.5
        chin = top + head
        upper = ys < top + 0.1 * (bottom - top)
        cx = float(np.median(xs[upper])) if upper.any() else float(np.median(xs))
        notes.append("No face found, so proportions were estimated from your full height.")
    if head < 8:
        return None

    def centre_run(y: int) -> Optional[Tuple[int, int]]:
        row = mask[y].astype(np.int8)
        diff = np.diff(np.concatenate(([0], row, [0])))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        if starts.size == 0:
            return None
        for s, e in zip(starts, ends):
            if s <= cx < e:
                return int(s), int(e) - 1
        dists = [min(abs(s - cx), abs(e - cx)) for s, e in zip(starts, ends)]
        i = int(np.argmin(dists))
        return int(starts[i]), int(ends[i]) - 1

    found: Dict[str, Optional[Dict[str, float]]] = {}
    for name, (a, b, mode) in BODY_BANDS.items():
        # The band must be (almost) fully inside the person, otherwise a photo
        # cropped at the hips would report the crop edge as the hip width.
        if bottom < chin + (b - 0.1) * head:
            found[name] = None
            continue
        y0, y1 = max(0, int(chin + a * head)), min(bottom, int(chin + b * head))
        rows = []
        for y in range(y0, y1 + 1):
            run = centre_run(y)
            if run is not None:
                rows.append((y, run[0], run[1]))
        if len(rows) < 5:
            found[name] = None
            continue
        widths = np.array([r[2] - r[1] + 1 for r in rows], dtype=np.float64)
        padded = np.pad(widths, 2, mode="edge")
        smooth = np.convolve(padded, np.ones(5) / 5.0, mode="valid")
        i = int(np.argmax(smooth) if mode == "max" else np.argmin(smooth))
        y, xl, xr = rows[i]
        found[name] = {"y": float(y), "xl": float(xl), "xr": float(xr), "width": float(smooth[i])}

    if found.get("shoulder") is None or found.get("waist") is None:
        return None

    s, wst = found["shoulder"]["width"], found["waist"]["width"]
    hp = found["hip"]["width"] if found.get("hip") else None
    confidence = 0.85 if face_box is not None else 0.55
    if hp is None:
        confidence -= 0.2
        notes.append("Hips are not visible - use a photo that reaches at least mid-thigh.")
    if wst >= 0.97 * s:
        confidence -= 0.15
        notes.append("Your waist reads as wide as your shoulders - arms may be touching your sides.")
    if face_box is not None and s < 1.6 * face_box[2]:
        confidence -= 0.2
        notes.append("Shoulders look narrow next to the head - the photo may be angled or cropped.")

    points: Dict[str, List[float]] = {}
    for name in ("shoulder", "waist", "hip"):
        lm = found.get(name)
        if lm is None:
            continue
        points[f"{name}_l"] = [round(lm["xl"] / img_w, 4), round(lm["y"] / img_h, 4)]
        points[f"{name}_r"] = [round(lm["xr"] / img_w, 4), round(lm["y"] / img_h, 4)]

    result = body_metrics(s, wst, hp)
    result.update({
        "detection_method": "auto_face_scaled" if face_box is not None else "auto_height_scaled",
        "confidence": round(max(0.1, min(0.95, confidence)), 2),
        "points": points,
        "head_px": round(float(head), 1),
        "notes": notes,
    })
    return result


def body_from_points(points: Dict[str, List[float]], image_size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    """Body metrics from manually tapped points (normalised, top-left origin)."""
    w, h = image_size

    def dist(a: str, b: str) -> Optional[float]:
        if a not in points or b not in points:
            return None
        (u1, v1), (u2, v2) = points[a], points[b]
        return math.hypot((u1 - u2) * w, (v1 - v2) * h)

    shoulder, waist, hip = dist("shoulder_l", "shoulder_r"), dist("waist_l", "waist_r"), dist("hip_l", "hip_r")
    if not shoulder or not waist:
        return None
    result = body_metrics(shoulder, waist, hip)
    result.update({
        "detection_method": "manual_tap",
        "confidence": 1.0,
        "points": {k: [round(float(v[0]), 4), round(float(v[1]), 4)] for k, v in points.items()
                   if k in BODY_POINT_KEYS},
        "notes": [],
    })
    return result


def sample_skin(bgr: "np.ndarray",
                face_box: Optional[Tuple[int, int, int, int]] = None,
                point: Optional[Tuple[float, float]] = None) -> Optional[Dict[str, Any]]:
    """Median CIELAB of skin pixels from both cheeks (or around a tapped point).

    Non-skin pixels (hair, eyes, background) are rejected with the YCrCb skin
    range; the darkest 15% (shadows) and brightest 10% (specular shine) are
    dropped before taking the median.
    """
    h, w = bgr.shape[:2]
    patches: List["np.ndarray"] = []
    if face_box is not None:
        fx, fy, fw, fh = face_box
        for x0, x1 in ((0.18, 0.40), (0.60, 0.82)):
            ya, yb = fy + int(0.48 * fh), fy + int(0.72 * fh)
            xa, xb = fx + int(x0 * fw), fx + int(x1 * fw)
            patches.append(bgr[max(0, ya):min(h, yb), max(0, xa):min(w, xb)].reshape(-1, 3))
        strict = True
    elif point is not None:
        px, py = int(point[0] * w), int(point[1] * h)
        r = max(4, int(min(h, w) * 0.025))
        patches.append(bgr[max(0, py - r):min(h, py + r), max(0, px - r):min(w, px + r)].reshape(-1, 3))
        strict = False
    else:
        return None
    pixels = np.concatenate([p for p in patches if p.size]) if any(p.size for p in patches) else None
    if pixels is None or pixels.shape[0] < 20:
        return None

    keep = _skin_mask(pixels.reshape(-1, 1, 3)).reshape(-1) > 0
    if keep.sum() >= max(50, 0.25 * pixels.shape[0]):
        pixels = pixels[keep]
    elif strict:
        return None

    lab = _bgr_to_lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    lo, hi = np.percentile(lab[:, 0], 15), np.percentile(lab[:, 0], 90)
    sel = (lab[:, 0] >= lo) & (lab[:, 0] <= hi)
    if sel.sum() >= 10:
        lab, pixels = lab[sel], pixels[sel]
    L, A, B = (float(v) for v in np.median(lab, axis=0))
    med_bgr = np.median(pixels, axis=0)
    return {
        "lab": [L, A, B],
        "hex": rgb01_to_hex(np.array([med_bgr[2], med_bgr[1], med_bgr[0]]) / 255.0),
        "pixels": int(pixels.shape[0]),
    }


def white_balance_offset(bgr: "np.ndarray", person_mask: Optional["np.ndarray"]) -> Optional[Tuple[float, float]]:
    """a*/b* colour cast of a bright, low-chroma backdrop (a wall under tinted
    light). None when the backdrop is genuinely coloured or too dark to judge."""
    if person_mask is None:
        return None
    bg = bgr[person_mask == 0]
    if bg.shape[0] < 1000:
        return None
    if bg.shape[0] > 20000:
        bg = bg[np.random.default_rng(0).choice(bg.shape[0], 20000, replace=False)]
    L, A, B = (float(v) for v in np.median(_bgr_to_lab(bg.reshape(-1, 1, 3)).reshape(-1, 3), axis=0))
    if L > 55.0 and math.hypot(A, B) < 30.0:
        return A, B
    return None


def skin_profile(raw: Dict[str, Any], wb: Optional[Tuple[float, float]], method: str) -> Dict[str, Any]:
    """Undertone, depth (ITA) and seasonal palette from a skin sample."""
    L, A, B = raw["lab"]
    if wb is not None:
        A -= 0.8 * wb[0]
        B -= 0.8 * wb[1]
    b_safe = max(B, 0.5)
    ita = math.degrees(math.atan((L - 50.0) / b_safe))
    hue = math.degrees(math.atan2(B, A))
    depth = "deep"
    for threshold, name in ITA_DEPTH_BANDS:
        if ita > threshold:
            depth = name
            break
    if hue < UNDERTONE_COOL_MAX_HUE:
        bias = "cool"
    elif hue > UNDERTONE_WARM_MIN_HUE:
        bias = "warm"
    else:
        bias = "neutral"
    seasons = {
        ("warm", False): "Spring", ("warm", True): "Autumn",
        ("cool", False): "Summer", ("cool", True): "Winter",
        ("neutral", False): "Soft Summer", ("neutral", True): "Soft Autumn",
    }
    deeper = depth in ("tan", "deep")
    season = seasons[(bias, deeper)]
    if depth == "deep":
        season = "Deep Autumn" if bias == "warm" else "Deep Winter"
    return {
        "sample_hex": raw["hex"],
        "lab": {"L": round(L, 2), "a": round(A, 2), "b": round(B, 2)},
        "ita_angle": round(ita, 1),
        "hue_angle": round(hue, 1),
        "undertone": "Deep" if depth == "deep" else bias.capitalize(),
        "undertone_bias": bias,
        "depth": depth,
        "seasonal_palette": season,
        "white_balance": "backdrop" if wb is not None else "none",
        "detection_method": method,
        "confidence": 0.8 if method == "auto_face" else 0.9,
    }


def build_palette(skin: Dict[str, Any]) -> Dict[str, Any]:
    """Flattering colours, neutrals and colours to go easy on."""
    bias = skin.get("undertone_bias", "neutral")
    rules = PALETTE_RULES.get(bias, PALETTE_RULES["neutral"])
    best, avoid = list(rules["best"]), list(rules["avoid"])
    if skin.get("depth") == "deep":
        best = list(dict.fromkeys(DEEP_BEST + best))
        avoid = list(dict.fromkeys(avoid + DEEP_AVOID))
    best = [n for n in best if n not in avoid][:10]

    def entries(names: List[str]) -> List[Dict[str, str]]:
        return [{"name": n, "hex": NAMED_COLOR_HEX[n]} for n in names if n in NAMED_COLOR_HEX]

    hues: List[List[float]] = []
    for n in best:
        r, g, b = hex_to_rgb01(NAMED_COLOR_HEX[n])
        hh, ss, _vv = colorsys.rgb_to_hsv(r, g, b)
        if ss > 0.2:
            hues.append([round((hh * 360 - 15) % 360, 1), round((hh * 360 + 15) % 360, 1)])
    return {
        "flattering_colors": entries(best),
        "best_neutrals": [NAMED_COLOR_HEX[n] for n in rules["neutrals"] if n in NAMED_COLOR_HEX],
        "best_neutral_names": [n for n in rules["neutrals"] if n in NAMED_COLOR_HEX],
        "avoid_colors": entries(avoid),
        "flattering_hues": hues,
        "basis": f"{bias} undertone, {skin.get('depth', 'medium')} depth",
    }


@dataclass
class BodyScan:
    """Outcome of one profile-photo analysis (nothing is saved yet)."""

    photo_bgr: "np.ndarray"
    image_size: Tuple[int, int]
    segmenter: str
    face_box: Optional[Tuple[int, int, int, int]]
    face_method: str
    body: Optional[Dict[str, Any]]
    skin: Optional[Dict[str, Any]]
    palette: Optional[Dict[str, Any]]
    processing_ms: int
    notes: List[str] = field(default_factory=list)

    def face_box_normalized(self) -> List[float]:
        if self.face_box is None:
            return []
        w, h = self.image_size
        x, y, bw, bh = self.face_box
        return [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]


class BodyProfiler:
    """Runs the biometric analysis on a portrait / full-body photo."""

    def __init__(self, engine: "VisionEngine", models_dir: str) -> None:
        self.engine = engine
        self.yunet_path = os.path.join(models_dir, FACE_YUNET_FILE)
        # Warm the face detector in the background so the first scan is fast.
        threading.Thread(target=_haar_cascade, daemon=True).start()

    def analyze(self, src_path: str, exif_orientation: int,
                progress: Callable[[str], None]) -> BodyScan:
        started = time.perf_counter()
        progress("Reading photo")
        bgr = limit_size(load_image_bgr(src_path, exif_orientation), PROFILE_WORK_SIDE)
        h, w = bgr.shape[:2]

        progress("Separating you from the background")
        alpha, seg_method, notes = self.engine.segment(bgr, None)  # never cloud
        mask: Optional["np.ndarray"] = None
        if seg_method != "none":
            mask = (alpha > 0.5).astype(np.uint8)

        progress("Finding your face")
        face_box, face_method = detect_face(bgr, mask, self.yunet_path)

        progress("Measuring proportions")
        body = measure_body(mask, face_box) if mask is not None else None
        if body is None:
            notes.append("Automatic measurement was not possible - tap the points instead.")

        progress("Reading skin tone")
        skin = None
        if face_box is not None:
            raw = sample_skin(bgr, face_box=face_box)
            if raw is not None:
                skin = skin_profile(raw, white_balance_offset(bgr, mask), "auto_face")
        if skin is None:
            notes.append("Skin tone could not be sampled automatically - tap your cheek instead.")
        palette = build_palette(skin) if skin is not None else None

        return BodyScan(
            photo_bgr=bgr,
            image_size=(w, h),
            segmenter=seg_method,
            face_box=face_box,
            face_method=face_method,
            body=body,
            skin=skin,
            palette=palette,
            processing_ms=int((time.perf_counter() - started) * 1000),
            notes=notes,
        )


# ===========================================================================
# STYLIST ENGINE
# Pure Python (no OpenCV needed): weather + trend ingestion (opt-in network),
# the combination matrix, Virtual Shuffle, capsule packing and gap finding.
# ===========================================================================

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_TTL_SECONDS = 3600
TREND_TTL_HOURS = 24
FEED_MAX_BYTES = 2_000_000
FEED_MAX_ENTRIES = 60

OCCASIONS: Tuple[str, ...] = ("Casual", "Formal", "DateNight", "Active", "Winter")
OCCASION_LABELS: Dict[str, str] = {"DateNight": "Date Night"}

HARMONY_LABELS: Dict[str, str] = {
    "monochromatic": "Monochrome column",
    "analogous": "Analogous blend",
    "complementary": "Complementary contrast",
    "split_complementary": "Split-complementary",
    "triadic": "Triadic pop",
    "neutral_anchor": "Neutral anchor",
    "neutral": "All-neutral classic",
    "clash": "Bold mix",
    "shuffle": "Avant-garde shuffle",
}

#: Total-score weights of the combination matrix.
SCORE_WEIGHTS: Dict[str, float] = {
    "harmony": 0.35, "skin_fit": 0.20, "weather_fit": 0.20, "occasion_fit": 0.15, "body_fit": 0.10,
}


class NetworkError(Exception):
    """An opt-in network feature could not reach its service."""


def _http_get(url: str, params: Optional[Dict[str, Any]] = None,
              headers: Optional[Dict[str, str]] = None, timeout: float = 10.0,
              max_bytes: int = FEED_MAX_BYTES) -> Tuple[int, Dict[str, str], bytes]:
    """Small GET helper: generic User-Agent, size cap, 304 passthrough."""
    import urllib.error
    import urllib.parse
    import urllib.request

    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    request_headers = {"User-Agent": f"WardrobeAI/{APP_VERSION}", "Accept": "*/*"}
    request_headers.update(headers or {})
    request = urllib.request.Request(url, headers=request_headers)
    context = _ssl_context() if url.startswith("https://") else None
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise NetworkError("the response is too large")
            return response.status, {k.lower(): v for k, v in response.headers.items()}, data
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return 304, {}, b""
        raise NetworkError(f"HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NetworkError(f"network error: {getattr(exc, 'reason', exc)}") from exc


def _on_wifi() -> bool:
    """True when Android's active network is Wi-Fi (always True off-Android)."""
    if not IS_ANDROID:
        return True
    try:
        from jnius import autoclass  # type: ignore[import-not-found]
        activity = autoclass("org.kivy.android.PythonActivity").mActivity
        Context = autoclass("android.content.Context")
        Caps = autoclass("android.net.NetworkCapabilities")
        manager = activity.getSystemService(Context.CONNECTIVITY_SERVICE)
        caps = manager.getNetworkCapabilities(manager.getActiveNetwork())
        return bool(caps is not None and caps.hasTransport(Caps.TRANSPORT_WIFI))
    except Exception:  # noqa: BLE001 - if we cannot tell, do not block the user
        return True


def network_allowed(storage: StorageManager) -> Tuple[bool, str]:
    if storage.get_setting("network.wifi_only", True) and not _on_wifi():
        return False, "Wi-Fi only is on and you are on mobile data."
    return True, ""


# --- Colour maths without OpenCV ------------------------------------------------

@functools.lru_cache(maxsize=4096)
def hex_to_lab(hex_code: str) -> Tuple[float, float, float]:
    """sRGB hex -> CIELAB (D65), pure Python (cached: wardrobes reuse colours)."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(v) for v in hex_to_rgb01(hex_code))
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883

    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    fx, fy, fz = f(x), f(y), f(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _delta_e(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _circ(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def item_color(item: Dict[str, Any]) -> Dict[str, Any]:
    """Primary colour of an item as {hex, name, h, s, v, lab, chromatic}."""
    colors = item.get("colors") or []
    if colors:
        c = colors[0]
        hsv = c.get("hsv", {})
        lab_d = c.get("lab")
        lab = (lab_d["L"], lab_d["a"], lab_d["b"]) if isinstance(lab_d, dict) else hex_to_lab(c["hex"])
        hex_code, name = c.get("hex", "#808080"), c.get("name", "")
        h, s, v = float(hsv.get("h", 0)), float(hsv.get("s", 0)), float(hsv.get("v", 0.5))
    else:
        hex_code = item.get("primary_hex") or "#808080"
        r, g, b = hex_to_rgb01(hex_code)
        hh, s, v = colorsys.rgb_to_hsv(r, g, b)
        h, lab, name = hh * 360.0, hex_to_lab(hex_code), ""
    family = item.get("color_family", "")
    chromatic = family != "neutral" and s >= 0.2 and v >= 0.15
    return {"hex": hex_code, "name": name, "h": h, "s": s, "v": v, "lab": lab, "chromatic": chromatic}


def item_label(item: Dict[str, Any]) -> str:
    return str(item.get("display_name") or item.get("category", "Item"))


def harmony_score(colors: List[Dict[str, Any]]) -> Tuple[float, str]:
    """Colour-wheel harmony of an outfit (0..1, rule).

    Neutrals (black, white, greys, beige, navy and low-saturation colours)
    pair with anything, so only chromatic hues are judged against the
    monochromatic / analogous / complementary / triadic rules.
    """
    hues = [c["h"] for c in colors if c["chromatic"]]
    if not hues:
        return 0.80, "neutral"
    reps: List[float] = []
    for h in hues:
        if not any(_circ(h, r) <= 15.0 for r in reps):
            reps.append(h)
    if len(reps) == 1:
        return (0.92, "monochromatic") if len(hues) > 1 else (0.86, "neutral_anchor")
    if len(reps) == 2:
        d = _circ(reps[0], reps[1])
        if d <= 40.0:
            return 0.84, "analogous"
        if d >= 150.0:
            return 0.70 + 0.25 * max(0.0, 1.0 - abs(180.0 - d) / 30.0), "complementary"
        if d >= 120.0:
            return 0.74, "split_complementary"
        return max(0.30, 0.60 - (d - 40.0) / 200.0), "clash"
    if len(reps) == 3:
        s = sorted(reps)
        gaps = [s[1] - s[0], s[2] - s[1], 360.0 - s[2] + s[0]]
        if all(abs(g - 120.0) <= 30.0 for g in gaps):
            return 0.85, "triadic"
        if max(_circ(a, b) for a in s for b in s) <= 60.0:
            return 0.78, "analogous"
        return 0.40, "clash"
    return 0.30, "clash"


# --- Layer physics ----------------------------------------------------------------

def _level(item: Dict[str, Any]) -> Optional[int]:
    level = item.get("layer_level")
    if level is None:
        level = CATEGORY_META.get(item.get("category", ""), {}).get("layer_level")
    return level


def valid_layer_chain(chain: List[Dict[str, Any]]) -> bool:
    """Layer physics for a stack of tops (inner first).

    * levels strictly increase: 0 T-shirt -> 1 shirt/sweater -> 2 hoodie ->
      3 jacket/coat, at most one item per level, so a hoodie can never sit
      under a shirt and two outer layers never clip;
    * the innermost piece is a base (level 0 or 1);
    * every outer piece must list each inner piece's level in can_layer_over.
    """
    if not chain:
        return False
    levels = [_level(t) for t in chain]
    if any(lv is None for lv in levels) or levels[0] not in (0, 1):
        return False
    if any(b <= a for a, b in zip(levels, levels[1:])):
        return False
    for i, outer in enumerate(chain[1:], start=1):
        allowed = outer.get("can_layer_over")
        if allowed is None:
            allowed = layers_under(outer.get("category", ""))
        if any(levels[j] not in allowed for j in range(i)):
            return False
    return True


# --- Context + per-dimension scores ---------------------------------------------------

@dataclass
class WeatherContext:
    temp: float                 # apparent ("feels like") temperature, deg C
    temp_min: float
    temp_max: float
    wet: bool
    snow: bool
    text: str
    icon: str
    source: str                 # live | cache | forecast | assumed
    place: str = ""
    date: str = ""


@dataclass
class StyleContext:
    occasion: str = "Casual"
    weather: Optional[WeatherContext] = None
    palette: Dict[str, Any] = field(default_factory=dict)
    undertone_bias: str = ""
    body_shape: Optional[str] = None
    recent_item_ids: frozenset = frozenset()


WEATHER_CODES: Dict[int, Tuple[str, str]] = {
    0: ("Clear", "weather-sunny"), 1: ("Mostly clear", "weather-partly-cloudy"),
    2: ("Partly cloudy", "weather-partly-cloudy"), 3: ("Overcast", "weather-cloudy"),
    45: ("Fog", "weather-fog"), 48: ("Fog", "weather-fog"),
    51: ("Light drizzle", "weather-rainy"), 53: ("Drizzle", "weather-rainy"), 55: ("Drizzle", "weather-rainy"),
    56: ("Freezing drizzle", "weather-snowy-rainy"), 57: ("Freezing drizzle", "weather-snowy-rainy"),
    61: ("Light rain", "weather-rainy"), 63: ("Rain", "weather-pouring"), 65: ("Heavy rain", "weather-pouring"),
    66: ("Freezing rain", "weather-snowy-rainy"), 67: ("Freezing rain", "weather-snowy-rainy"),
    71: ("Light snow", "weather-snowy"), 73: ("Snow", "weather-snowy"), 75: ("Heavy snow", "weather-snowy-heavy"),
    77: ("Snow grains", "weather-snowy"), 80: ("Showers", "weather-rainy"), 81: ("Showers", "weather-pouring"),
    82: ("Violent showers", "weather-pouring"), 85: ("Snow showers", "weather-snowy"),
    86: ("Snow showers", "weather-snowy-heavy"), 95: ("Thunderstorm", "weather-lightning"),
    96: ("Thunderstorm", "weather-lightning-rainy"), 99: ("Thunderstorm", "weather-lightning-rainy"),
}


def describe_weather(code: Optional[int]) -> Tuple[str, str, bool, bool]:
    """(text, icon, wet, snow) for an Open-Meteo WMO weather code."""
    code = int(code) if code is not None else -1
    text, icon = WEATHER_CODES.get(code, ("Weather", "weather-cloudy"))
    wet = 51 <= code <= 67 or 80 <= code <= 82 or code >= 95
    snow = 71 <= code <= 77 or code in (85, 86)
    return text, icon, wet, snow


def _outfit_warmth(items: List[Dict[str, Any]]) -> float:
    total = 0.0
    for it in items:
        slot = it.get("slot") or CATEGORY_META.get(it.get("category", ""), {}).get("slot")
        if slot in ("top", "bottom", "full_body"):
            total += float(it.get("warmth_index") or CATEGORY_META.get(it.get("category", ""), {}).get("warmth", 2))
    return total


def _target_warmth(temp: float) -> float:
    for threshold, need in ((26.0, 2.0), (20.0, 3.0), (14.0, 4.5), (8.0, 6.0), (0.0, 7.5)):
        if temp >= threshold:
            return need
    return 9.0


def weather_fit(items: List[Dict[str, Any]], weather: Optional[WeatherContext]) -> Tuple[bool, float, List[str]]:
    """Hard climate filters plus a 0..1 warmth match."""
    if weather is None:
        return True, 0.7, []
    t = weather.temp
    tops = [i for i in items if (i.get("slot") or CATEGORY_META.get(i.get("category", ""), {}).get("slot")) == "top"]
    levels = {_level(i) for i in tops}
    cats = {i.get("category") for i in items}
    has_outer, has_mid = 3 in levels, 2 in levels
    # Hard rules - hide clearly wrong outfits.
    if t >= 24.0 and (len(tops) > 1 or any(float(i.get("warmth_index") or 0) >= 3 for i in tops)):
        return False, 0.0, []
    if t < 18.0 and "Shorts" in cats:
        return False, 0.0, []
    if t < 10.0 and tops and not (has_outer or has_mid):
        return False, 0.0, []
    if t < 0.0 and "Coat" not in cats and not (has_outer and has_mid):
        return False, 0.0, []
    if weather.wet and any({"delicate", "suede", "silk"} & set(i.get("fabric_tags") or []) for i in items):
        return False, 0.0, []

    score = max(0.0, 1.0 - abs(_outfit_warmth(items) - _target_warmth(t)) / 5.0)
    reasons = [f"Built for {round(t)}°C"]
    if weather.wet:
        if has_outer:
            score += 0.10
            reasons.append("Outer layer for the rain")
        else:
            score -= 0.15
        if cats & {"Shorts", "Skirt", "Dress"}:
            score -= 0.10
    if weather.snow and "Coat" in cats:
        score += 0.05
    return True, max(0.0, min(1.0, score)), reasons


def occasion_fit(items: List[Dict[str, Any]], occasion: str) -> Tuple[bool, float]:
    main = [i for i in items if (i.get("slot") or "") not in ("footwear", "accessory")]
    if not all(occasion in (i.get("occasions") or []) for i in main):
        return False, 0.0
    if occasion == "Winter" and not any(_level(i) in (2, 3) for i in main):
        return False, 0.0
    formalities = [float(i.get("formality") or 2) for i in main]
    spread = max(formalities) - min(formalities) if formalities else 0.0
    score = 1.0 - max(0.0, spread - 1.0) * 0.2
    if occasion == "Formal" and formalities and sum(formalities) / len(formalities) < 3.0:
        score -= 0.25
    return True, max(0.0, min(1.0, score))


def skin_fit(roles: List[Tuple[Dict[str, Any], float]], ctx: StyleContext) -> Tuple[float, List[str]]:
    """How well colours near the face suit the measured palette."""
    palette = ctx.palette or {}
    good = [hex_to_lab(c["hex"]) for c in palette.get("flattering_colors", [])]
    good += [hex_to_lab(h) for h in palette.get("best_neutrals", [])]
    bad = [hex_to_lab(c["hex"]) for c in palette.get("avoid_colors", [])]
    if not good:
        return 0.6, []
    acc = total = 0.0
    reasons: List[str] = []
    for item, weight in roles:
        col = item_color(item)
        d_good = min(_delta_e(col["lab"], g) for g in good)
        s = 1.0 if d_good < 18.0 else (0.8 if d_good < 30.0 else 0.55)
        if bad and min(_delta_e(col["lab"], b) for b in bad) < 14.0:
            s = min(s, 0.25)
        fam = item.get("color_family", "")
        if (ctx.undertone_bias == "warm" and fam in ("warm", "earth")) or \
                (ctx.undertone_bias == "cool" and fam == "cool"):
            s += 0.05
        acc += min(1.0, s) * weight
        total += weight
        if weight >= 0.8 and d_good < 18.0 and not reasons:
            bias = ctx.undertone_bias or "skin"
            reasons.append(f"{col['name'] or 'This colour'} flatters your {bias} undertone")
    return (acc / total if total else 0.6), reasons


def body_fit(tops: List[Dict[str, Any]], lower: Optional[Dict[str, Any]], shape: Optional[str]) -> Tuple[float, List[str]]:
    """Proportion-balancing guidance per body shape (general style rules)."""
    if not shape or not tops:
        return 0.65, []
    top_v = item_color(tops[-1])["v"]
    low_v = item_color(lower)["v"] if lower else top_v
    layered = len(tops) >= 2
    if shape == "inverted_triangle":
        return (0.85, ["Darker top balances broader shoulders"]) if top_v <= low_v else (0.5, [])
    if shape == "triangle":
        ok = top_v >= low_v or 3 in {_level(t) for t in tops}
        return (0.85, ["Lighter or structured top balances the hips"]) if ok else (0.5, [])
    if shape == "rectangle":
        ok = layered or abs(top_v - low_v) > 0.25
        return (0.85, ["Layers add shape to a straight frame"]) if ok else (0.55, [])
    if shape == "oval":
        same = lower is not None and _delta_e(item_color(tops[-1])["lab"], item_color(lower)["lab"]) < 25.0
        ok = same or 3 in {_level(t) for t in tops}
        return (0.85, ["Long column of colour elongates"]) if ok else (0.55, [])
    if shape == "hourglass":
        return 0.8, []
    return 0.65, []


# --- Combination matrix ------------------------------------------------------------------

def _slot(item: Dict[str, Any]) -> str:
    return str(item.get("slot") or CATEGORY_META.get(item.get("category", ""), {}).get("slot", ""))


def _outfit_id(item_ids: List[str]) -> str:
    import hashlib
    return hashlib.sha1("|".join(sorted(item_ids)).encode("utf-8")).hexdigest()[:16]


def _single_item_prescore(item: Dict[str, Any], ctx: StyleContext) -> float:
    score, _ = skin_fit([(item, 1.0)], ctx)
    return score - (0.1 if item.get("item_id") in ctx.recent_item_ids else 0.0)


def build_top_chains(tops: List[Dict[str, Any]], ctx: StyleContext, caps: Tuple[int, int, int, int] = (5, 5, 3, 3)) -> List[List[Dict[str, Any]]]:
    by_level: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: [], 3: []}
    for t in tops:
        lv = _level(t)
        if lv in by_level:
            by_level[lv].append(t)
    for lv in by_level:
        by_level[lv].sort(key=lambda i: _single_item_prescore(i, ctx), reverse=True)
        by_level[lv] = by_level[lv][:caps[lv]]
    chains: List[List[Dict[str, Any]]] = []
    for base in by_level[0] + by_level[1]:
        seconds: List[Optional[Dict[str, Any]]] = [None] + (by_level[1][:3] if _level(base) == 0 else [])
        for second in seconds:
            for mid in [None] + by_level[2]:
                for outer in [None] + by_level[3]:
                    chain = [x for x in (base, second, mid, outer) if x is not None]
                    if valid_layer_chain(chain):
                        chains.append(chain)
    return chains


def _evaluate(items: List[Dict[str, Any]], tops: List[Dict[str, Any]], lower: Optional[Dict[str, Any]],
              ctx: StyleContext, shuffle: bool) -> Optional[Dict[str, Any]]:
    reasons: List[str] = []
    if shuffle:
        cols = [item_color(i) for i in items]
        hue_spread = max((_circ(a["h"], b["h"]) for a in cols for b in cols if a["chromatic"] and b["chromatic"]),
                         default=0.0) / 180.0
        value_contrast = max(c["v"] for c in cols) - min(c["v"] for c in cols)
        total = 0.6 * hue_spread + 0.4 * value_contrast
        return {"items": items, "rule": "shuffle", "total": total,
                "scores": {"contrast": round(total, 3), "total": round(total, 3)},
                "reasons": ["High contrast", "Rules off - break the routine"]}

    ok_w, w_score, w_reasons = weather_fit(items, ctx.weather)
    if not ok_w:
        return None
    ok_o, o_score = occasion_fit(items, ctx.occasion)
    if not ok_o:
        return None
    h_score, rule = harmony_score([item_color(i) for i in items])
    face_roles: List[Tuple[Dict[str, Any], float]] = []
    for idx, t in enumerate(tops):
        face_roles.append((t, 1.0 if idx == len(tops) - 1 else 0.7))
    if lower is not None:
        face_roles.append((lower, 1.0 if _slot(lower) == "full_body" else 0.3))
    s_score, s_reasons = skin_fit(face_roles, ctx)
    b_score, b_reasons = body_fit(tops, lower, ctx.body_shape)
    total = (SCORE_WEIGHTS["harmony"] * h_score + SCORE_WEIGHTS["skin_fit"] * s_score
             + SCORE_WEIGHTS["weather_fit"] * w_score + SCORE_WEIGHTS["occasion_fit"] * o_score
             + SCORE_WEIGHTS["body_fit"] * b_score)
    fresh_penalty = min(0.15, 0.05 * sum(1 for i in items if i.get("item_id") in ctx.recent_item_ids))
    total -= fresh_penalty
    reasons = [HARMONY_LABELS[rule]] + s_reasons + w_reasons + b_reasons
    if fresh_penalty:
        reasons.append("Includes something worn recently")
    return {"items": items, "rule": rule, "total": total, "reasons": reasons,
            "scores": {"harmony": round(h_score, 3), "skin_fit": round(s_score, 3),
                       "weather_fit": round(w_score, 3), "occasion_fit": round(o_score, 3),
                       "body_fit": round(b_score, 3), "total": round(total, 3)}}


def generate_outfits(items: List[Dict[str, Any]], ctx: StyleContext, limit: int = 10,
                     shuffle: bool = False, seed: Optional[int] = None,
                     diversify: bool = True) -> List[Dict[str, Any]]:
    """Rank complete outfits. ``shuffle`` drops occasion/weather/harmony rules
    (layer physics always applies) and favours high contrast with randomness.
    ``diversify`` skips near-duplicates so a swipe deck stays varied."""
    import random
    rng = random.Random(seed)
    active = [i for i in items if i.get("status", "active") == "active"]
    tops = [i for i in active if _slot(i) == "top"]
    bottoms = [i for i in active if _slot(i) == "bottom"]
    dresses = [i for i in active if _slot(i) == "full_body"]
    shoes = [i for i in active if _slot(i) == "footwear"]

    if shuffle:
        tops = rng.sample(tops, len(tops))
    chains = build_top_chains(tops, ctx, caps=(6, 6, 4, 4) if shuffle else (5, 5, 3, 3))
    outers = [[t] for t in tops if _level(t) == 3]

    candidates: List[Dict[str, Any]] = []
    for chain in chains:
        for bottom in bottoms:
            ev = _evaluate(chain + [bottom], chain, bottom, ctx, shuffle)
            if ev is not None:
                candidates.append(ev)
    for dress in dresses:
        for layer in [[]] + outers:
            ev = _evaluate(layer + [dress], layer, dress, ctx, shuffle)
            if ev is not None:
                candidates.append(ev)
    if shuffle:
        for c in candidates:
            c["total"] += rng.uniform(0.0, 0.25)
    candidates.sort(key=lambda c: c["total"], reverse=True)

    picked: List[Dict[str, Any]] = []
    for cand in candidates:
        ids = {i["item_id"] for i in cand["items"]}
        if diversify and any(len(ids & {i["item_id"] for i in p["items"]}) >= max(2, len(ids) - 1)
                             for p in picked):
            continue  # too similar to an outfit already picked - keep the deck varied
        picked.append(cand)
        if len(picked) >= limit:
            break

    outfits = []
    now = utc_now_iso()
    for cand in picked:
        chosen = list(cand["items"])
        if shoes:
            base_cols = [item_color(i) for i in chosen]
            best_shoe = max(shoes, key=lambda s: harmony_score(base_cols + [item_color(s)])[0]
                            + (rng.random() * 0.2 if shuffle else 0.0))
            chosen.append(best_shoe)
        order = {"top": 0, "full_body": 1, "bottom": 2, "footwear": 3, "accessory": 4}
        chosen.sort(key=lambda i: (order.get(_slot(i), 5), _level(i) or 0))
        ids = [i["item_id"] for i in chosen]
        outfits.append({
            "outfit_id": _outfit_id(ids),
            "item_ids": ids,
            "scores": cand["scores"],
            "harmony_rule": cand["rule"],
            "title": HARMONY_LABELS.get(cand["rule"], "Outfit"),
            "occasion": ctx.occasion,
            "source": "shuffle" if shuffle else "suggested",
            "reasons": list(dict.fromkeys(cand["reasons"])),
            "weather": ({"temp": round(ctx.weather.temp, 1), "text": ctx.weather.text}
                        if ctx.weather is not None else None),
            "saved": False,
            "created_at": now,
        })
    return outfits


# --- Weather (Open-Meteo, opt-in) ---------------------------------------------------------

def geocode_city(name: str, base_url: str = OPEN_METEO_GEOCODE_URL) -> Dict[str, Any]:
    _status, _h, data = _http_get(base_url, {"name": name, "count": 1, "language": "en", "format": "json"})
    try:
        results = json.loads(data.decode("utf-8")).get("results") or []
    except (ValueError, UnicodeDecodeError) as exc:
        raise NetworkError("unreadable geocoding response") from exc
    if not results:
        raise NetworkError(f"no place called '{name}' was found")
    r = results[0]
    label = ", ".join(x for x in (r.get("name"), r.get("admin1"), r.get("country")) if x)
    # Two decimals (~1 km) is plenty for weather and shares less precision.
    return {"name": label, "lat": round(float(r["latitude"]), 2), "lon": round(float(r["longitude"]), 2)}


def fetch_forecast(lat: float, lon: float, days: int = 16,
                   base_url: str = OPEN_METEO_FORECAST_URL) -> Dict[str, Any]:
    params = {
        "latitude": round(lat, 2), "longitude": round(lon, 2),
        "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "daily": ("weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,"
                  "apparent_temperature_min,precipitation_probability_max,precipitation_sum"),
        "timezone": "auto", "forecast_days": max(1, min(16, days)),
    }
    _status, _h, data = _http_get(base_url, params)
    try:
        j = json.loads(data.decode("utf-8"))
        cur, daily = j.get("current") or {}, j.get("daily") or {}
        days_out = []
        for i, date in enumerate(daily.get("time") or []):
            def pick(key: str) -> Any:
                values = daily.get(key) or []
                return values[i] if i < len(values) else None
            days_out.append({
                "date": date, "code": pick("weather_code"),
                "t_max": pick("temperature_2m_max"), "t_min": pick("temperature_2m_min"),
                "feels_max": pick("apparent_temperature_max"), "feels_min": pick("apparent_temperature_min"),
                "precip_prob": pick("precipitation_probability_max"), "precip_sum": pick("precipitation_sum"),
            })
        return {
            "fetched_at": time.time(),
            "timezone": j.get("timezone"),
            "current": {
                "temp": cur.get("temperature_2m"), "feels": cur.get("apparent_temperature"),
                "precip": cur.get("precipitation"), "code": cur.get("weather_code"),
                "wind": cur.get("wind_speed_10m"),
            },
            "daily": days_out,
        }
    except (ValueError, UnicodeDecodeError, TypeError) as exc:
        raise NetworkError("unreadable forecast response") from exc


def weather_from_current(fc: Dict[str, Any], place: str, source: str) -> Optional[WeatherContext]:
    cur = fc.get("current") or {}
    if cur.get("temp") is None:
        return None
    today = (fc.get("daily") or [{}])[0]
    text, icon, wet, snow = describe_weather(cur.get("code"))
    wet = wet or float(cur.get("precip") or 0) > 0.2 or float(today.get("precip_prob") or 0) >= 60
    feels = float(cur.get("feels") if cur.get("feels") is not None else cur["temp"])
    return WeatherContext(
        temp=feels,
        temp_min=float(today.get("t_min") if today.get("t_min") is not None else feels),
        temp_max=float(today.get("t_max") if today.get("t_max") is not None else feels),
        wet=wet, snow=snow, text=text, icon=icon, source=source, place=place,
        date=str(today.get("date") or ""))


def weather_from_day(day: Dict[str, Any], place: str) -> WeatherContext:
    text, icon, wet, snow = describe_weather(day.get("code"))
    wet = wet or float(day.get("precip_prob") or 0) >= 60 or float(day.get("precip_sum") or 0) >= 2.0
    hi = day.get("feels_max") if day.get("feels_max") is not None else day.get("t_max")
    lo = day.get("feels_min") if day.get("feels_min") is not None else day.get("t_min")
    hi, lo = float(hi if hi is not None else 18.0), float(lo if lo is not None else 10.0)
    # Dress for the daytime, weighted towards the cooler morning/evening.
    temp = lo + 0.6 * (hi - lo)
    return WeatherContext(temp=temp, temp_min=lo, temp_max=hi, wet=wet, snow=snow, text=text,
                          icon=icon, source="forecast", place=place, date=str(day.get("date", "")))


def assumed_weather(date: str = "", place: str = "") -> WeatherContext:
    return WeatherContext(temp=16.0, temp_min=11.0, temp_max=21.0, wet=False, snow=False,
                          text="Mild (no forecast)", icon="weather-partly-cloudy", source="assumed",
                          place=place, date=date)


WEATHER_CACHE_REL = "cache/weather_cache.json"


def load_weather(storage: StorageManager, allow_network: bool,
                 base_url: str = OPEN_METEO_FORECAST_URL) -> Tuple[Optional[WeatherContext], str]:
    """Current weather for the profile location: fresh cache -> network (if
    opted in) -> stale cache. Never touches the network when opted out."""
    if not storage.get_setting("network.weather_enabled", False):
        return None, "Weather is off"
    loc = storage.load("profile").get("location") or {}
    if loc.get("lat") is None or loc.get("lon") is None:
        return None, "Set your city to use the weather"
    place = str(loc.get("city") or "")
    cache_path = storage.path(WEATHER_CACHE_REL)
    cached = read_json(cache_path, None)
    same_place = isinstance(cached, dict) and cached.get("lat") == loc["lat"] and cached.get("lon") == loc["lon"]
    if same_place and time.time() - float(cached.get("fetched_at", 0)) < WEATHER_TTL_SECONDS:
        return weather_from_current(cached, place, "cache"), ""
    if allow_network:
        allowed, why = network_allowed(storage)
        if allowed:
            try:
                fc = fetch_forecast(float(loc["lat"]), float(loc["lon"]), base_url=base_url)
                fc.update({"lat": loc["lat"], "lon": loc["lon"]})
                atomic_write_json(cache_path, fc)
                return weather_from_current(fc, place, "live"), ""
            except NetworkError as exc:
                why = str(exc)
        if same_place:
            return weather_from_current(cached, place, "cache"), f"Using saved weather ({why})"
        return None, why
    if same_place:
        return weather_from_current(cached, place, "cache"), ""
    return None, ""


# --- Trend feeds (RSS / Atom, opt-in, 24 h cache) ------------------------------------------

_COLOR_WORDS: Dict[str, str] = {name.lower(): name for name, _hex in NAMED_COLORS}
_COLOR_WORDS.update({
    "black": "Jet Black", "white": "Optic White", "grey": "Heather Grey", "gray": "Heather Grey",
    "charcoal": "Charcoal Grey", "navy": "Navy Blue", "beige": "Sand Beige", "sand": "Sand Beige",
    "stone": "Sand Beige", "camel": "Camel Brown", "brown": "Mocha Brown", "chocolate": "Dark Chocolate",
    "olive": "Olive Green", "sage": "Sage Green", "mint": "Mint Green", "green": "Emerald Green",
    "emerald": "Emerald Green", "blue": "Royal Blue", "cobalt": "Royal Blue", "denim": "Denim Blue",
    "light blue": "Powder Blue", "red": "Crimson Red", "wine": "Burgundy", "maroon": "Burgundy",
    "pink": "Baby Pink", "blush": "Dusty Rose", "rose": "Dusty Rose", "lilac": "Lavender",
    "purple": "Royal Purple", "yellow": "Sunflower Yellow", "butter yellow": "Sunflower Yellow",
    "mustard": "Mustard Yellow", "orange": "Bright Orange", "rust": "Rust Orange", "ecru": "Cream",
    "gold": "Antique Gold",
})
_GARMENT_WORDS: Dict[str, str] = {
    "t-shirt": "T-Shirt", "tshirt": "T-Shirt", "tee": "T-Shirt", "polo": "T-Shirt",
    "shirt": "Shirt", "button-down": "Shirt", "blouse": "Shirt", "oxford": "Shirt",
    "sweater": "Sweater", "jumper": "Sweater", "knit": "Sweater", "knitwear": "Sweater",
    "cardigan": "Sweater", "turtleneck": "Sweater", "hoodie": "Hoodie", "sweatshirt": "Hoodie",
    "jacket": "Jacket", "blazer": "Jacket", "bomber": "Jacket", "shacket": "Jacket",
    "coat": "Coat", "trench": "Coat", "trench coat": "Coat", "overcoat": "Coat", "parka": "Coat", "puffer": "Coat",
    "jeans": "Jeans", "trousers": "Trouser", "trouser": "Trouser", "slacks": "Trouser",
    "pants": "Pants", "chinos": "Pants", "cargos": "Pants", "cargo pants": "Pants", "joggers": "Pants",
    "shorts": "Shorts", "skirt": "Skirt", "dress": "Dress", "gown": "Dress",
    "sneakers": "Shoes", "trainers": "Shoes", "boots": "Shoes", "loafers": "Shoes", "shoes": "Shoes",
    "heels": "Shoes",
}


def _alternation(words: Dict[str, str]) -> str:
    import re
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


_TREND_PATTERN = None


def extract_components(text: str) -> List[Tuple[str, str]]:
    """'camel wool coat with black jeans' -> [("Coat", "Camel Brown"), ("Jeans", "Jet Black")]."""
    import re
    global _TREND_PATTERN
    if _TREND_PATTERN is None:
        _TREND_PATTERN = re.compile(
            rf"\b(?P<color>{_alternation(_COLOR_WORDS)})\b(?:[\s-]+[a-z]+){{0,2}}?[\s-]+"
            rf"(?P<garment>{_alternation(_GARMENT_WORDS)})\b", re.IGNORECASE)
    found: List[Tuple[str, str]] = []
    for mt in _TREND_PATTERN.finditer(text.lower()):
        comp = (_GARMENT_WORDS[mt.group("garment").lower()], _COLOR_WORDS[mt.group("color").lower()])
        if comp not in found:
            found.append(comp)
    return found


def parse_feed(data: bytes) -> List[Dict[str, Any]]:
    """Entries from an RSS 2.0 or Atom feed: title, summary, published (epoch)."""
    import html
    import re
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise NetworkError(f"not a valid RSS/Atom feed ({exc})") from exc

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    def clean(text: Optional[str]) -> str:
        return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()

    def when(text: Optional[str]) -> float:
        if not text:
            return time.time()
        text = text.strip()
        try:
            return parsedate_to_datetime(text).timestamp()
        except (TypeError, ValueError, IndexError):
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return time.time()

    entries = []
    for node in root.iter():
        if local(node.tag) not in ("item", "entry"):
            continue
        fields: Dict[str, str] = {}
        cats: List[str] = []
        for child in node:
            name = local(child.tag)
            if name == "category":
                cats.append(child.get("term") or child.text or "")
            elif name in ("title", "description", "summary", "content", "pubdate", "published", "updated"):
                fields.setdefault(name, child.text or "")
        entries.append({
            "title": clean(fields.get("title"))[:200],
            "summary": clean(fields.get("description") or fields.get("summary") or fields.get("content"))[:600],
            "categories": [clean(c)[:40] for c in cats if c][:8],
            "published": when(fields.get("pubdate") or fields.get("published") or fields.get("updated")),
        })
        if len(entries) >= FEED_MAX_ENTRIES:
            break
    return entries


def build_trends(sources: List[Dict[str, Any]], now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Turn feed entries into trend combinations of >= 2 garment slots,
    weighted by how often they appear and how recent they are."""
    import hashlib
    now = now or time.time()
    combos: Dict[str, Dict[str, Any]] = {}
    for src in sources:
        for entry in src.get("entries") or []:
            comps = extract_components(" ".join([entry.get("title", ""), entry.get("summary", ""),
                                                 " ".join(entry.get("categories") or [])]))
            by_slot: Dict[str, Tuple[str, str]] = {}
            for category, color in comps:
                slot = CATEGORY_META.get(category, {}).get("slot", "")
                by_slot.setdefault(slot, (category, color))
            if len(by_slot) < 2:
                continue
            chosen = sorted(by_slot.values())[:4]
            key = hashlib.sha1(repr(chosen).encode("utf-8")).hexdigest()[:12]
            age_days = max(0.0, (now - float(entry.get("published") or now)) / 86400.0)
            weight = math.exp(-age_days / 14.0)
            trend = combos.setdefault(key, {
                "trend_id": key,
                "title": entry.get("title") or " + ".join(f"{c} {g}" for g, c in chosen),
                "source_id": src.get("id", ""),
                "fetched_at": utc_now_iso(),
                "palette": [NAMED_COLOR_HEX.get(c, "#808080") for _g, c in chosen],
                "tags": list(entry.get("categories") or [])[:5],
                "components": [{
                    "slot": CATEGORY_META.get(g, {}).get("slot", ""), "category": g, "color_name": c,
                    "hex_target": NAMED_COLOR_HEX.get(c, "#808080"), "tolerance_deltaE": 22.0,
                } for g, c in chosen],
                "popularity_score": 0.0,
            })
            trend["popularity_score"] = round(trend["popularity_score"] + weight, 3)
    return sorted(combos.values(), key=lambda t: t["popularity_score"], reverse=True)[:30]


def item_matches_component(item: Dict[str, Any], comp: Dict[str, Any]) -> bool:
    groups = {"Pants": "long", "Trouser": "long"}
    cat, want = item.get("category"), comp.get("category")
    if cat != want and not (groups.get(cat) and groups.get(cat) == groups.get(want)):
        return False
    names = {c.get("name") for c in item.get("colors") or []}
    if comp.get("color_name") in names:
        return True
    return _delta_e(item_color(item)["lab"], hex_to_lab(comp["hex_target"])) <= float(comp.get("tolerance_deltaE", 22.0))


def match_trends(items: List[Dict[str, Any]], trends: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    active = [i for i in items if i.get("status", "active") == "active"]
    matches = []
    for trend in trends:
        matched, missing = [], []
        for comp in trend.get("components", []):
            owner = next((i for i in active if item_matches_component(i, comp)), None)
            if owner is not None:
                matched.append(owner["item_id"])
            else:
                missing.append(comp)
        total = len(trend.get("components", [])) or 1
        matches.append({"trend_id": trend["trend_id"], "match_pct": round(100.0 * len(matched) / total),
                        "matched_item_ids": matched, "missing_components": missing, "notified": False})
    return matches


def find_gaps(items: List[Dict[str, Any]], trends: List[Dict[str, Any]],
              matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trends missing exactly one piece -> wishlist candidates, ranked by how
    many new outfits that one piece would unlock."""
    by_id = {t["trend_id"]: t for t in trends}
    active = [i for i in items if i.get("status", "active") == "active"]
    gaps: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        trend = by_id.get(m["trend_id"])
        if trend is None or len(m["missing_components"]) != 1 or len(trend["components"]) < 2:
            continue
        comp = m["missing_components"][0]
        target = {"h": 0.0, "s": 0.0, "v": 0.5, "chromatic": False, "lab": hex_to_lab(comp["hex_target"])}
        r, g, b = hex_to_rgb01(comp["hex_target"])
        hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
        target.update({"h": hh * 360.0, "s": ss, "v": vv,
                       "chromatic": comp["color_name"] not in NEUTRAL_COLOR_NAMES and ss >= 0.2})
        partner_slots = {"top": {"bottom"}, "bottom": {"top"}, "full_body": {"top", "footwear"},
                         "footwear": {"top", "bottom", "full_body"}}.get(comp["slot"], {"top", "bottom"})
        unlocks = sum(1 for i in active if _slot(i) in partner_slots
                      and harmony_score([item_color(i), target])[0] >= 0.6)
        key = f"{comp['category']}|{comp['color_name']}"
        wish_id = _outfit_id([key])
        current = gaps.get(wish_id)
        priority = float(trend.get("popularity_score", 0))
        if current is None or priority > current["priority"]:
            gaps[wish_id] = {
                "wish_id": wish_id, "category": comp["category"], "color_name": comp["color_name"],
                "hex_target": comp["hex_target"], "reason": "trend_gap", "trend_id": trend["trend_id"],
                "trend_title": trend.get("title", ""), "unlocks_outfits": unlocks,
                "priority": round(priority, 3), "added_at": utc_now_iso(), "acquired": False,
            }
    return sorted(gaps.values(), key=lambda w: (w["unlocks_outfits"], w["priority"]), reverse=True)


def sync_trends(storage: StorageManager, force: bool = False) -> Tuple[Dict[str, Any], str]:
    """Fetch enabled feeds (conditional GET), rebuild trends, match them
    against the wardrobe and refresh wishlist gaps. Opt-in only."""
    doc = storage.load("trends")
    if not storage.get_setting("network.trends_enabled", False):
        return doc, "Trend feeds are off"
    last = doc.get("last_sync")
    ttl = float(doc.get("ttl_hours") or TREND_TTL_HOURS) * 3600.0
    fresh = isinstance(last, (int, float)) and time.time() - float(last) < ttl
    messages: List[str] = []
    if not fresh or force:
        allowed, why = network_allowed(storage)
        if not allowed:
            messages.append(why)
        else:
            for src in doc.get("sources") or []:
                if not src.get("enabled", True):
                    continue
                headers = {}
                if src.get("etag"):
                    headers["If-None-Match"] = src["etag"]
                if src.get("last_modified"):
                    headers["If-Modified-Since"] = src["last_modified"]
                try:
                    status, resp_headers, data = _http_get(src["url"], headers=headers, timeout=12.0)
                    if status == 304:
                        src["last_status"] = "not modified"
                    else:
                        src["entries"] = parse_feed(data)
                        src["etag"] = resp_headers.get("etag", "")
                        src["last_modified"] = resp_headers.get("last-modified", "")
                        src["last_status"] = f"ok · {len(src['entries'])} posts"
                except NetworkError as exc:
                    src["last_status"] = f"error: {exc}"
                    messages.append(f"{src.get('url', 'feed')}: {exc}")
                src["last_fetch"] = utc_now_iso()
            doc["last_sync"] = time.time()

    items = list(storage.load("items").get("items", {}).values())
    doc["trends"] = build_trends(doc.get("sources") or [])
    doc["matches"] = match_trends(items, doc["trends"])
    storage.save("trends", doc)
    refresh_wishlist(storage, items, doc)
    return doc, "; ".join(messages)


def refresh_wishlist(storage: StorageManager, items: List[Dict[str, Any]], trends_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Merge trend gaps into wishlist.json; respect dismissals; mark gaps
    that an owned item now fills as acquired."""
    gaps = find_gaps(items, trends_doc.get("trends") or [], trends_doc.get("matches") or [])

    def _apply(doc: Dict[str, Any]) -> None:
        wishes = [w for w in doc.get("wishes") or [] if isinstance(w, dict)]
        dismissed = set(doc.get("dismissed") or [])
        by_id = {w.get("wish_id"): w for w in wishes}
        for gap in gaps:
            if gap["wish_id"] in dismissed:
                continue
            if gap["wish_id"] in by_id:
                by_id[gap["wish_id"]].update({k: gap[k] for k in ("unlocks_outfits", "priority", "trend_title", "trend_id")})
            else:
                wishes.append(gap)
        live_ids = {g["wish_id"] for g in gaps}
        for w in wishes:
            if w.get("reason") == "trend_gap" and w.get("wish_id") not in live_ids and not w.get("acquired"):
                comp = {"category": w.get("category"), "color_name": w.get("color_name"),
                        "hex_target": w.get("hex_target", "#808080"), "tolerance_deltaE": 22.0}
                if any(item_matches_component(i, comp) for i in items):
                    w["acquired"] = True
        doc["wishes"] = wishes
        doc.setdefault("dismissed", [])

    return storage.update("wishlist", _apply)


# --- Capsule packing wizard ----------------------------------------------------------------

def _count_variations(items: List[Dict[str, Any]], ctx: StyleContext, min_score: float = 0.55) -> int:
    return sum(1 for o in generate_outfits(items, ctx, limit=200, diversify=False)
               if o["scores"]["total"] >= min_score)


def plan_capsule(items: List[Dict[str, Any]], day_weather: List[WeatherContext],
                 occasions: List[str], palette: Optional[Dict[str, Any]] = None,
                 undertone_bias: str = "", body_shape: Optional[str] = None) -> Dict[str, Any]:
    """Greedy capsule: add the piece that unlocks the most new outfit
    variations until there is a different outfit for every day (or adding
    more stops helping), then plan one outfit per day for that day's weather."""
    occasions = [o for o in occasions if o in OCCASIONS] or ["Casual"]
    days = len(day_weather)
    temps = [w.temp for w in day_weather] or [16.0]
    typical = WeatherContext(temp=sorted(temps)[len(temps) // 2], temp_min=min(temps), temp_max=max(temps),
                             wet=any(w.wet for w in day_weather), snow=any(w.snow for w in day_weather),
                             text="Trip", icon="bag-suitcase", source="assumed")
    ctx = StyleContext(occasion=occasions[0], weather=typical, palette=palette or {},
                       undertone_bias=undertone_bias, body_shape=body_shape)
    active = [i for i in items if i.get("status", "active") == "active"
              and (any(o in (i.get("occasions") or []) for o in occasions) or _slot(i) == "footwear")]
    clothing = [i for i in active if _slot(i) in ("top", "bottom", "full_body")]
    max_items = min(len(clothing), max(4, math.ceil(days * 0.7) + 2))

    chosen: List[Dict[str, Any]] = []
    notes: List[str] = []
    needs_outer = min(temps) < 10.0 or typical.wet
    if needs_outer:
        outers = [i for i in clothing if _level(i) == 3]
        if outers:
            chosen.append(max(outers, key=lambda i: (float(i.get("warmth_index") or 0), _single_item_prescore(i, ctx))))
        else:
            notes.append("No jacket or coat in your wardrobe for the colder / wet days.")

    variations = 0
    while len(chosen) < max_items:
        best, best_gain = None, 0
        for cand in clothing:
            if cand in chosen:
                continue
            gain = _count_variations(chosen + [cand], ctx) - variations
            if gain > best_gain or (gain == best_gain and best is not None and gain > 0
                                    and _single_item_prescore(cand, ctx) > _single_item_prescore(best, ctx)):
                best, best_gain = cand, gain
        if best is None:
            # Nothing adds an outfit on its own yet (e.g. no top+bottom pair):
            # seed with the most versatile top or bottom still missing.
            have = {_slot(i) for i in chosen}
            need = "bottom" if "top" in have else "top"
            pool = [i for i in clothing if _slot(i) == need and i not in chosen]
            if not pool or len(chosen) >= max_items:
                break
            best = max(pool, key=lambda i: _single_item_prescore(i, ctx))
        chosen.append(best)
        variations = _count_variations(chosen, ctx)
        if variations >= days and len(chosen) >= 3:
            break

    shoes = [i for i in active if _slot(i) == "footwear"]
    if shoes:
        cols = [item_color(i) for i in chosen]
        chosen.append(max(shoes, key=lambda s: harmony_score(cols + [item_color(s)])[0]))

    plan, used = [], {}
    for idx, weather in enumerate(day_weather):
        day_ctx = StyleContext(occasion=occasions[idx % len(occasions)], weather=weather, palette=palette or {},
                               undertone_bias=undertone_bias, body_shape=body_shape)
        # diversify=False: a small capsule's outfits naturally share pieces;
        # the day planner itself spreads them out via the ``used`` counter.
        options = generate_outfits(chosen, day_ctx, limit=24, diversify=False)
        if not options:
            day_ctx.occasion = "Casual"
            options = generate_outfits(chosen, day_ctx, limit=24, diversify=False)
        pick = min(options, key=lambda o: (used.get(o["outfit_id"], 0), -o["scores"]["total"])) if options else None
        if pick is not None:
            used[pick["outfit_id"]] = used.get(pick["outfit_id"], 0) + 1
        plan.append({"day": idx + 1, "date": weather.date, "weather": {"temp": round(weather.temp),
                     "text": weather.text, "wet": weather.wet}, "outfit": pick})
    if any(p["outfit"] is None for p in plan):
        notes.append("Some days have no complete outfit for that weather - consider adding a layer.")
    return {
        "capsule_item_ids": [i["item_id"] for i in chosen],
        "variation_count": _count_variations(chosen, ctx),
        "outfit_plan": plan,
        "unique_outfits_used": len(used),
        "notes": notes,
    }


# --- Wear logging (feeds freshness + cost-per-wear) ------------------------------------------

def recent_item_ids(storage: StorageManager, days: int = 2) -> frozenset:
    from datetime import date, timedelta
    entries = storage.load("wear_log").get("entries") or {}
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    ids = set()
    for day, worn in entries.items():
        if day >= cutoff:
            for entry in worn:
                ids.update(entry.get("item_ids") or [])
    return frozenset(ids)


def log_wear(storage: StorageManager, outfit: Dict[str, Any]) -> int:
    """Record an outfit as worn today; update wear counts and cost-per-wear."""
    from datetime import date
    today = date.today().isoformat()

    def _log(doc: Dict[str, Any]) -> None:
        entries = doc.setdefault("entries", {})
        day = entries.setdefault(today, [])
        if not any(e.get("outfit_id") == outfit["outfit_id"] for e in day):
            day.append({"outfit_id": outfit["outfit_id"], "item_ids": list(outfit["item_ids"]),
                        "occasion": outfit.get("occasion"), "weather_snapshot": outfit.get("weather"),
                        "rating": None})

    storage.update("wear_log", _log)
    updated = 0

    def _items(doc: Dict[str, Any]) -> None:
        nonlocal updated
        for iid in outfit["item_ids"]:
            item = (doc.get("items") or {}).get(iid)
            if item is None or item.get("last_worn") == today:
                continue
            item["wear_count"] = int(item.get("wear_count") or 0) + 1
            item["last_worn"] = today
            price = item.get("purchase_price")
            item["cost_per_wear"] = round(float(price) / item["wear_count"], 2) if price else None
            item["updated_at"] = utc_now_iso()
            updated += 1

    storage.update("items", _items)

    def _outfits(doc: Dict[str, Any]) -> None:
        stored = doc.setdefault("outfits", {}).setdefault(outfit["outfit_id"], dict(outfit))
        stored["last_worn"] = today
        stored["wear_count"] = int(stored.get("wear_count") or 0) + 1

    storage.update("outfits", _outfits)
    return updated


# ===========================================================================
# ANDROID GALLERY PICKER (system picker intent - no storage permission needed)
# ===========================================================================

class AndroidImagePicker:
    """Opens the Android system image picker and copies the chosen image into
    the app's private cache as a JPEG (downsampled, EXIF orientation read).

    Callbacks are always delivered on the Kivy main thread:
        on_picked(path, exif_orientation), on_cancel(), on_error(message)
    """

    REQUEST_CODE = 0x5CA1
    RESULT_OK = -1

    def __init__(self, dest_dir: str, on_picked: Callable[[str, int], None],
                 on_cancel: Callable[[], None], on_error: Callable[[str], None]) -> None:
        self.dest_dir = dest_dir
        self.on_picked = on_picked
        self.on_cancel = on_cancel
        self.on_error = on_error

    def open(self) -> None:
        from android import activity  # type: ignore[import-not-found]
        from jnius import autoclass  # type: ignore[import-not-found]

        Intent = autoclass("android.content.Intent")
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.setType("image/*")
        intent.addCategory(Intent.CATEGORY_OPENABLE)
        activity.bind(on_activity_result=self._on_activity_result)
        PythonActivity.mActivity.startActivityForResult(intent, self.REQUEST_CODE)

    def _on_activity_result(self, request_code: int, result_code: int, intent: Any) -> None:
        if request_code != self.REQUEST_CODE:
            return
        from android import activity  # type: ignore[import-not-found]
        activity.unbind(on_activity_result=self._on_activity_result)
        if result_code != self.RESULT_OK or intent is None or intent.getData() is None:
            Clock.schedule_once(lambda _dt: self.on_cancel(), 0)
            return
        uri = intent.getData()
        threading.Thread(target=self._copy_worker, args=(uri,), daemon=True).start()

    def _copy_worker(self, uri: Any) -> None:
        try:
            path, orientation = self._copy_uri(uri)
            Clock.schedule_once(lambda _dt: self.on_picked(path, orientation), 0)
        except Exception as exc:  # noqa: BLE001 - surface any Java/IO failure to the UI
            Logger.exception(f"{LOG_TAG}: gallery copy failed")
            message = f"Could not open that image: {exc}"
            Clock.schedule_once(lambda _dt: self.on_error(message), 0)

    def _copy_uri(self, uri: Any) -> Tuple[str, int]:
        from jnius import autoclass  # type: ignore[import-not-found]

        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        BitmapFactory = autoclass("android.graphics.BitmapFactory")
        BitmapOptions = autoclass("android.graphics.BitmapFactory$Options")
        CompressFormat = autoclass("android.graphics.Bitmap$CompressFormat")
        FileOutputStream = autoclass("java.io.FileOutputStream")
        ExifInterface = autoclass("android.media.ExifInterface")

        resolver = PythonActivity.mActivity.getContentResolver()

        # 1) Read bounds only, to choose a power-of-two downsample factor.
        bounds = BitmapOptions()
        bounds.inJustDecodeBounds = True
        stream = resolver.openInputStream(uri)
        try:
            BitmapFactory.decodeStream(stream, None, bounds)
        finally:
            stream.close()
        sample = 1
        longest = max(int(bounds.outWidth), int(bounds.outHeight))
        while longest // (sample * 2) >= MAX_WORK_SIDE:
            sample *= 2

        # 2) EXIF orientation (BitmapFactory ignores it).
        orientation = 1
        try:
            stream = resolver.openInputStream(uri)
            try:
                orientation = int(ExifInterface(stream).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION, 1))
            finally:
                stream.close()
        except Exception:  # noqa: BLE001 - PNGs etc. have no EXIF
            orientation = 1

        # 3) Decode downsampled and re-encode as JPEG in our private cache.
        opts = BitmapOptions()
        opts.inSampleSize = sample
        stream = resolver.openInputStream(uri)
        try:
            bitmap = BitmapFactory.decodeStream(stream, None, opts)
        finally:
            stream.close()
        if bitmap is None:
            raise VisionError("The selected file is not a supported image.")

        os.makedirs(self.dest_dir, exist_ok=True)
        dest = os.path.join(self.dest_dir, f"pick_{uuid.uuid4().hex}.jpg")
        out = FileOutputStream(dest)
        try:
            bitmap.compress(CompressFormat.JPEG, 95, out)
            out.flush()
        finally:
            out.close()
            bitmap.recycle()
        return dest, orientation


# ===========================================================================
# DESIGN SYSTEM - colour tokens for light and dark themes
# ===========================================================================

#: Every UI colour comes from these tokens; KV binds to ``app.c_*`` so a theme
#: switch re-colours the whole window with one animated transition.
#: 8-digit hex = RRGGBBAA.
PALETTES: Dict[str, Dict[str, str]] = {
    # Minimal luxury: warm milk canvas, pitch-black type and accents,
    # ultra-thin hairlines. The neon "laser" is reserved for AI moments.
    "Light": {
        "bg": "#F9F9FB",
        "surface": "#FFFFFF",
        "surface_alt": "#F3F3F6",
        "frame": "#F1F1F4",
        "text": "#000000",
        "muted": "#6E6E73",
        "accent": "#000000",
        "accent_soft": "#0000000D",
        "border": "#E4E4E9",
        "on_accent": "#FFFFFF",
        "overlay": "#08090BE6",
        "laser": "#00F2FE",
        "spot": "#FFFFFF",
    },
    # Obsidian midnight canvas, stark white type, neon-cyan highlights.
    "Dark": {
        "bg": "#090D10",
        "surface": "#10161B",
        "surface_alt": "#151C22",
        "frame": "#0D1317",
        "text": "#FFFFFF",
        "muted": "#8B95A1",
        "accent": "#00F2FE",
        "accent_soft": "#00F2FE1C",
        "border": "#FFFFFF17",
        "on_accent": "#00181A",
        "overlay": "#000000EB",
        "laser": "#00F2FE",
        "spot": "#FFFFFF",
    },
}

#: KivyMD's own primary palette per theme (text-field focus lines, dialogs).
MD_PRIMARY: Dict[str, str] = {"Light": "Gray", "Dark": "Cyan"}


# ===========================================================================
# UI LAYER
# ===========================================================================

KV = """
#:import dp kivy.metrics.dp
#:import FadeTransition kivy.uix.screenmanager.FadeTransition
#:import get_color_from_hex kivy.utils.get_color_from_hex
#:import Window kivy.core.window.Window
#:import md_icons kivymd.icon_definitions.md_icons


<Surface>:
    canvas.before:
        Color:
            rgba: self.bg
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [self.radius_v]
        Color:
            rgba: self.line
        Line:
            rounded_rectangle: (self.x + self.line_w / 2.0, self.y + self.line_w / 2.0, max(0, self.width - self.line_w), max(0, self.height - self.line_w), self.radius_v)
            width: max(1.0, self.line_w)


<Txt@Label>:
    color: app.c_text
    font_name: "Roboto"
    font_size: "13sp"
    halign: "left"
    valign: "middle"
    text_size: self.width, None
    size_hint_y: None
    height: self.texture_size[1]


<TxtW@Label>:
    color: app.c_text
    font_name: "Roboto"
    font_size: "13sp"
    size_hint: None, None
    size: self.texture_size


<Ico@Label>:
    icon: ""
    text: md_icons.get(self.icon, "")
    font_name: "Icons"
    font_size: "20sp"
    color: app.c_text
    size_hint: None, None
    size: self.texture_size


<IconTap>:
    text: md_icons.get(self.icon, "")
    font_name: "Icons"
    color: app.c_text
    opacity: 0.6 if self.state == "down" else 1




# ---------------------------------------------------------------- primitives

# ---- luxury typography: Roboto Light display, Roboto Medium small caps

<LuxDisplay@MDLabel>:
    font_name: "RobotoLight"
    font_size: "30sp"
    adaptive_height: True
    theme_text_color: "Custom"
    text_color: app.c_text


<LuxTitle@MDLabel>:
    font_name: "RobotoLight"
    font_size: "21sp"
    adaptive_height: True
    theme_text_color: "Custom"
    text_color: app.c_text


<LuxKicker@MDLabel>:
    font_style: "Overline"
    font_name: "RobotoMedium"
    font_size: "10sp"
    adaptive_height: True
    theme_text_color: "Custom"
    text_color: app.c_muted


<LuxBody@MDLabel>:
    font_name: "Roboto"
    font_size: "13sp"
    adaptive_height: True
    theme_text_color: "Custom"
    text_color: app.c_muted


<PillButton>:
    size_hint_y: None
    height: dp(50)
    radius: [dp(25)]
    elevation: 0
    md_bg_color: app.c_accent if root.filled else (0, 0, 0, 0)
    line_color: app.c_accent if root.filled else app.c_border
    line_width: dp(0.8)
    opacity: 0.78 if self.state == "down" else 1

    MDAnchorLayout:
        MDBoxLayout:
            adaptive_size: True
            spacing: dp(8)

            MDIcon:
                icon: root.icon
                font_size: "19sp"
                size_hint: None, None
                size: dp(20), dp(20)
                halign: "center"
                valign: "middle"
                pos_hint: {"center_y": 0.5}
                theme_text_color: "Custom"
                text_color: app.c_on_accent if root.filled else app.c_text

            MDLabel:
                text: root.text
                font_style: "Button"
                font_name: "RobotoMedium"
                font_size: "12sp"
                adaptive_size: True
                pos_hint: {"center_y": 0.5}
                theme_text_color: "Custom"
                text_color: app.c_on_accent if root.filled else app.c_text


<ChipButton>:
    size_hint: None, None
    height: dp(34)
    width: chip_row.minimum_width + dp(30)
    padding: dp(15), 0
    radius_v: dp(17)
    bg: app.c_accent if root.selected else (0, 0, 0, 0)
    line: app.c_accent if (root.selected or root.pulse > 0.01) else app.c_border
    line_w: dp(0.8) + dp(2.2) * root.pulse
    opacity: 0.75 if self.state == "down" else 1

    BoxLayout:
        id: chip_row
        size_hint_x: None
        width: self.minimum_width
        spacing: dp(6)

        Ico:
            icon: root.icon or "blank"
            font_size: "16sp"
            size: (dp(16), dp(16)) if root.icon else (0, 0)
            opacity: 1 if root.icon else 0
            color: app.c_on_accent if root.selected else app.c_text
            pos_hint: {"center_y": 0.5}

        TxtW:
            text: root.text
            font_name: "RobotoMedium" if root.selected else "Roboto"
            font_size: "13sp"
            color: app.c_on_accent if root.selected else app.c_text
            pos_hint: {"center_y": 0.5}


<SwatchPill>:
    size_hint: None, None
    height: dp(34)
    width: swatch_row.minimum_width + dp(24)
    padding: dp(8), 0, dp(14), 0
    radius_v: dp(17)
    bg: 0, 0, 0, 0
    line: app.c_border

    BoxLayout:
        id: swatch_row
        size_hint_x: None
        width: self.minimum_width
        spacing: dp(8)

        ColorDot:
            rgba: get_color_from_hex(root.hex_color)
            pos_hint: {"center_y": 0.5}

        TxtW:
            text: root.label
            font_size: "13sp"
            pos_hint: {"center_y": 0.5}

        TxtW:
            text: root.detail
            font_size: "11sp"
            color: app.c_muted
            pos_hint: {"center_y": 0.5}


# Dynamic chip rows (choice_grid, swatch_row, provider_chips) use Kivy's plain
# StackLayout on purpose: KivyMD 1.2's themable remove_widget scans every
# theme observer per child, which made clearing ~10 chips take ~350 ms.

<Collapsible>:
    orientation: "vertical"
    size_hint_y: None
    height: self.minimum_height if self.shown else 0
    opacity: 1 if self.shown else 0


<SectionTitle@MDLabel>:
    font_style: "Overline"
    font_name: "RobotoMedium"
    font_size: "10sp"
    adaptive_height: True
    theme_text_color: "Custom"
    text_color: app.c_muted


<Toast>:
    size_hint: None, None
    height: dp(44)
    width: toast_row.minimum_width + dp(36)
    radius: [dp(22)]
    elevation: 0
    padding: dp(18), 0
    opacity: 0
    md_bg_color: app.c_text
    pos_hint: {"center_x": 0.5, "top": 0.985}

    MDBoxLayout:
        id: toast_row
        adaptive_width: True
        spacing: dp(8)

        MDIcon:
            icon: root.icon
            font_size: "20sp"
            size_hint: None, None
            size: dp(22), dp(22)
            halign: "center"
            valign: "middle"
            pos_hint: {"center_y": 0.5}
            theme_text_color: "Custom"
            text_color: app.c_bg

        MDLabel:
            text: root.text
            font_name: "RobotoMedium"
            font_size: "13sp"
            adaptive_width: True
            pos_hint: {"center_y": 0.5}
            theme_text_color: "Custom"
            text_color: app.c_bg


# ---------------------------------------------------------------- scanner

<CaptureFrame>:
    size_hint_y: None
    height: self.width * 1.02
    radius: [dp(28)]
    elevation: 0
    padding: 0
    md_bg_color: app.c_frame
    line_color: app.c_border
    line_width: dp(0.8)

    canvas.after:
        Color:
            rgba: app.c_accent if not root.has_image else (app.c_accent[0], app.c_accent[1], app.c_accent[2], 0.45)
        Line:
            points: [self.x + dp(18), self.top - dp(48), self.x + dp(18), self.top - dp(18), self.x + dp(48), self.top - dp(18)]
            width: dp(1.6)
            cap: "round"
            joint: "round"
        Line:
            points: [self.right - dp(48), self.top - dp(18), self.right - dp(18), self.top - dp(18), self.right - dp(18), self.top - dp(48)]
            width: dp(1.6)
            cap: "round"
            joint: "round"
        Line:
            points: [self.x + dp(18), self.y + dp(48), self.x + dp(18), self.y + dp(18), self.x + dp(48), self.y + dp(18)]
            width: dp(1.6)
            cap: "round"
            joint: "round"
        Line:
            points: [self.right - dp(48), self.y + dp(18), self.right - dp(18), self.y + dp(18), self.right - dp(18), self.y + dp(48)]
            width: dp(1.6)
            cap: "round"
            joint: "round"

    MDRelativeLayout:

        # Soft "studio spotlight" behind the cut-out so dark garments keep
        # their contrast on the charcoal frame (and pop on the light one).
        canvas.before:
            Color:
                rgba: (1, 1, 1, (0.14 if app.theme_cls.theme_style == "Dark" else 0.75) if root.has_image else 0)
            Ellipse:
                pos: self.width * 0.5 - min(self.width, self.height) * 0.44, self.height * 0.5 - min(self.width, self.height) * 0.44
                size: min(self.width, self.height) * 0.88, min(self.width, self.height) * 0.88

        MDBoxLayout:
            orientation: "vertical"
            adaptive_height: True
            spacing: dp(12)
            padding: dp(24), 0
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            opacity: 0 if root.has_image else 1

            MDCard:
                size_hint: None, None
                size: dp(88), dp(88)
                radius: [dp(44)]
                elevation: 0
                pos_hint: {"center_x": 0.5}
                md_bg_color: app.c_accent_soft

                MDAnchorLayout:
                    MDIcon:
                        size_hint: None, None
                        size: self.font_size + dp(8), self.font_size + dp(8)
                        icon: "tshirt-crew-outline"
                        font_size: "42sp"
                        halign: "center"
                        valign: "middle"
                        theme_text_color: "Custom"
                        text_color: app.c_accent

            MDLabel:
                text: "Your item appears here"
                font_style: "Subtitle1"
                bold: True
                halign: "center"
                adaptive_height: True
                theme_text_color: "Custom"
                text_color: app.c_text

            MDLabel:
                text: "Camera or upload · background removed automatically"
                font_style: "Caption"
                halign: "center"
                adaptive_height: True
                theme_text_color: "Custom"
                text_color: app.c_muted

        Image:
            source: root.image_source
            fit_mode: "contain"
            nocache: True
            size_hint: 0.84, 0.84
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            opacity: 1 if root.has_image else 0

        MDCard:
            size_hint: None, None
            height: dp(28)
            width: pill_row.minimum_width + dp(22)
            radius: [dp(14)]
            elevation: 0
            padding: dp(10), 0, dp(12), 0
            pos_hint: {"x": 0.06, "top": 0.95}
            opacity: 1 if root.has_image and root.pill_text else 0
            md_bg_color: app.c_surface
            line_color: app.c_border
            line_width: dp(0.8)

            MDBoxLayout:
                id: pill_row
                adaptive_width: True
                spacing: dp(4)

                MDIcon:
                    icon: "lightning-bolt"
                    font_size: "14sp"
                    size_hint: None, None
                    size: dp(16), dp(16)
                    halign: "center"
                    valign: "middle"
                    pos_hint: {"center_y": 0.5}
                    theme_text_color: "Custom"
                    text_color: app.c_accent

                MDLabel:
                    text: root.pill_text
                    font_style: "Caption"
                    bold: True
                    adaptive_width: True
                    pos_hint: {"center_y": 0.5}
                    theme_text_color: "Custom"
                    text_color: app.c_text


<ConfirmBadge>:
    size_hint_y: None
    height: dp(84)

    canvas.before:
        Color:
            rgba: app.c_accent[0], app.c_accent[1], app.c_accent[2], 0.10 + 0.32 * root.glow
        RoundedRectangle:
            pos: 0, 0
            size: self.size
            radius: [dp(42)]

    MDCard:
        size_hint: None, None
        size: root.width - dp(10), root.height - dp(10)
        pos: dp(5), dp(5)
        radius: [dp(37)]
        elevation: 0
        padding: dp(10), dp(8), dp(20), dp(8)
        spacing: dp(12)
        md_bg_color: app.c_accent
        opacity: 0.85 if root.state == "down" else 1

        MDCard:
            size_hint: None, None
            size: dp(50), dp(50)
            radius: [dp(25)]
            elevation: 0
            pos_hint: {"center_y": 0.5}
            md_bg_color: app.c_on_accent[0], app.c_on_accent[1], app.c_on_accent[2], 0.14

            MDAnchorLayout:
                MDIcon:
                    size_hint: None, None
                    size: self.font_size + dp(8), self.font_size + dp(8)
                    icon: "check-decagram"
                    font_size: "28sp"
                    halign: "center"
                    valign: "middle"
                    theme_text_color: "Custom"
                    text_color: app.c_on_accent

        MDBoxLayout:
            orientation: "vertical"
            adaptive_height: True
            spacing: dp(2)
            pos_hint: {"center_y": 0.5}

            MDLabel:
                text: root.text
                font_style: "Subtitle1"
                bold: True
                adaptive_height: True
                theme_text_color: "Custom"
                text_color: app.c_on_accent

            MDLabel:
                text: root.subtext
                font_style: "Caption"
                adaptive_height: True
                theme_text_color: "Custom"
                text_color: app.c_on_accent[0], app.c_on_accent[1], app.c_on_accent[2], 0.78

        MDIcon:
            icon: "arrow-right"
            font_size: "24sp"
            size_hint: None, None
            size: dp(26), dp(26)
            halign: "center"
            valign: "middle"
            pos_hint: {"center_y": 0.5}
            theme_text_color: "Custom"
            text_color: app.c_on_accent


<ScanOverlay>:
    auto_dismiss: False
    background_color: 0, 0, 0, 0
    overlay_color: app.c_overlay
    size_hint: 1, 1

    MDAnchorLayout:

        MDBoxLayout:
            orientation: "vertical"
            size_hint: None, None
            width: min(root.width - dp(56), dp(320))
            height: self.minimum_height
            spacing: dp(18)

            # ---- photo frame with neon corner brackets and a glowing laser
            MDRelativeLayout:
                size_hint: None, None
                width: min(root.width - dp(96), dp(250))
                height: self.width * 1.25
                pos_hint: {"center_x": 0.5}

                canvas.before:
                    Color:
                        rgba: 1, 1, 1, 0.04
                    RoundedRectangle:
                        pos: 0, 0
                        size: self.size
                        radius: [dp(28)]

                Image:
                    source: root.source_image
                    fit_mode: "contain"
                    nocache: True
                    size_hint: 0.86, 0.86
                    pos_hint: {"center_x": 0.5, "center_y": 0.5}
                    opacity: 0.55

                Widget:
                    canvas:
                        # soft bloom
                        Color:
                            rgba: app.c_laser[0], app.c_laser[1], app.c_laser[2], 0.07
                        Rectangle:
                            pos: dp(10), self.height * root.scan_pos - dp(34)
                            size: self.width - dp(20), dp(68)
                        # halo
                        Color:
                            rgba: app.c_laser[0], app.c_laser[1], app.c_laser[2], 0.20
                        Rectangle:
                            pos: dp(10), self.height * root.scan_pos - dp(10)
                            size: self.width - dp(20), dp(20)
                        # core beam
                        Color:
                            rgba: app.c_laser[0], app.c_laser[1], app.c_laser[2], 0.95
                        Rectangle:
                            pos: dp(10), self.height * root.scan_pos - dp(1)
                            size: self.width - dp(20), dp(2)
                        # corner brackets (pulse with the glow)
                        Color:
                            rgba: app.c_laser[0], app.c_laser[1], app.c_laser[2], 0.45 + 0.5 * root.glow
                        Line:
                            points: [dp(4), self.height - dp(34), dp(4), self.height - dp(4), dp(34), self.height - dp(4)]
                            width: dp(1.6)
                            cap: "round"
                            joint: "round"
                        Line:
                            points: [self.width - dp(34), self.height - dp(4), self.width - dp(4), self.height - dp(4), self.width - dp(4), self.height - dp(34)]
                            width: dp(1.6)
                            cap: "round"
                            joint: "round"
                        Line:
                            points: [dp(4), dp(34), dp(4), dp(4), dp(34), dp(4)]
                            width: dp(1.6)
                            cap: "round"
                            joint: "round"
                        Line:
                            points: [self.width - dp(34), dp(4), self.width - dp(4), dp(4), self.width - dp(4), dp(34)]
                            width: dp(1.6)
                            cap: "round"
                            joint: "round"

            # ---- title row
            MDBoxLayout:
                size_hint: None, None
                width: title_label.width + dp(34)
                height: dp(30)
                spacing: dp(10)
                pos_hint: {"center_x": 0.5}

                MDSpinner:
                    size_hint: None, None
                    size: dp(20), dp(20)
                    pos_hint: {"center_y": 0.5}
                    color: app.c_laser
                    line_width: dp(1.6)
                    active: root.active

                MDLabel:
                    id: title_label
                    text: root.title
                    font_name: "RobotoLight"
                    font_size: "21sp"
                    adaptive_width: True
                    pos_hint: {"center_y": 0.5}
                    theme_text_color: "Custom"
                    text_color: 1, 1, 1, 1

            MDLabel:
                text: root.stage_text
                font_name: "Roboto"
                font_size: "13sp"
                halign: "center"
                adaptive_height: True
                theme_text_color: "Custom"
                text_color: 1, 1, 1, 0.7

            # ---- progress bar + metrics
            Widget:
                size_hint_y: None
                height: dp(3)
                canvas:
                    Color:
                        rgba: 1, 1, 1, 0.12
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [dp(1.5)]
                    Color:
                        rgba: app.c_laser
                    RoundedRectangle:
                        pos: self.pos
                        size: max(dp(3), self.width * root.progress), self.height
                        radius: [dp(1.5)]

            MDBoxLayout:
                size_hint_y: None
                height: dp(16)

                MDLabel:
                    text: root.percent_text
                    font_style: "Overline"
                    font_name: "RobotoMedium"
                    font_size: "10sp"
                    theme_text_color: "Custom"
                    text_color: 1, 1, 1, 0.75

                MDLabel:
                    text: root.elapsed_text
                    font_style: "Overline"
                    font_name: "RobotoMedium"
                    font_size: "10sp"
                    halign: "right"
                    theme_text_color: "Custom"
                    text_color: app.c_laser

            MDLabel:
                text: root.privacy_text
                font_style: "Overline"
                font_name: "RobotoMedium"
                font_size: "10sp"
                halign: "center"
                adaptive_height: True
                theme_text_color: "Custom"
                text_color: 1, 1, 1, 0.5


<ScannerZone>:
    md_bg_color: app.c_bg

    MDScrollView:
        do_scroll_x: False
        bar_width: 0

        MDBoxLayout:
            orientation: "vertical"
            adaptive_height: True
            padding: dp(22), dp(10), dp(22), dp(112)
            spacing: dp(16)

            MDBoxLayout:
                orientation: "vertical"
                adaptive_height: True
                spacing: dp(4)

                LuxKicker:
                    text: "AI RESULT" if root.has_result else "ADD TO WARDROBE"
                    text_color: app.c_accent

                LuxDisplay:
                    text: "Looks right?" if root.has_result else "Scan a garment"

                MDLabel:
                    text: root.confidence_text if root.has_result else "Lay it flat on a plain, contrasting surface and fill most of the frame."
                    font_style: "Body2"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_muted

            CaptureFrame:
                image_source: root.image_source
                has_image: root.has_result
                pill_text: root.frame_badge

            # ---------- idle: capture actions + engine card
            Collapsible:
                shown: not root.has_result
                spacing: dp(14)

                MDBoxLayout:
                    size_hint_y: None
                    height: dp(52)
                    spacing: dp(12)

                    PillButton:
                        icon: "camera-outline"
                        text: "Camera"
                        filled: True
                        on_release: root.open_camera()

                    PillButton:
                        icon: "image-plus"
                        text: "Upload photo"
                        filled: False
                        on_release: root.open_gallery()

                MDCard:
                    orientation: "horizontal"
                    size_hint_y: None
                    height: self.minimum_height
                    radius: [dp(20)]
                    elevation: 0
                    padding: dp(14)
                    spacing: dp(12)
                    md_bg_color: app.c_surface
                    line_color: app.c_border
                    line_width: dp(0.8)

                    MDCard:
                        size_hint: None, None
                        size: dp(40), dp(40)
                        radius: [dp(20)]
                        elevation: 0
                        pos_hint: {"center_y": 0.5}
                        md_bg_color: app.c_accent_soft

                        MDAnchorLayout:
                            MDIcon:
                                size_hint: None, None
                                size: self.font_size + dp(8), self.font_size + dp(8)
                                icon: root.engine_icon
                                font_size: "22sp"
                                halign: "center"
                                valign: "middle"
                                theme_text_color: "Custom"
                                text_color: app.c_accent

                    MDBoxLayout:
                        orientation: "vertical"
                        adaptive_height: True
                        spacing: dp(2)
                        pos_hint: {"center_y": 0.5}

                        MDLabel:
                            text: root.engine_title
                            font_style: "Subtitle2"
                            bold: True
                            adaptive_height: True
                            theme_text_color: "Custom"
                            text_color: app.c_text

                        MDLabel:
                            text: root.engine_status
                            font_style: "Caption"
                            adaptive_height: True
                            theme_text_color: "Custom"
                            text_color: app.c_muted

            # ---------- result: confirmation badge + suggestions
            Collapsible:
                shown: root.has_result
                spacing: dp(14)

                ConfirmBadge:
                    id: confirm_badge
                    text: root.confirm_text
                    subtext: "One tap saves it to your wardrobe"
                    on_release: root.confirm()

                SectionTitle:
                    text: "NOT QUITE? PICK THE RIGHT TYPE"

                StackLayout:
                    id: choice_grid
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

                SectionTitle:
                    text: "DETECTED COLOURS"

                StackLayout:
                    id: swatch_row
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

                MDLabel:
                    id: meta_label
                    font_style: "Caption"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_muted

                MDBoxLayout:
                    size_hint_y: None
                    height: dp(52)
                    spacing: dp(12)

                    PillButton:
                        icon: "close"
                        text: "Discard"
                        filled: False
                        on_release: root.discard()

                    PillButton:
                        icon: "camera-retake-outline"
                        text: "Retake"
                        filled: False
                        on_release: root.retake()

    Toast:
        id: toast


# ---------------------------------------------------------------- other zones

<ZoneCard>:
    orientation: "vertical"
    padding: dp(24)
    spacing: dp(12)
    size_hint_y: None
    height: self.minimum_height
    radius: [dp(24)]
    elevation: 0
    md_bg_color: app.c_surface
    line_color: app.c_border
    line_width: dp(0.8)

    MDCard:
        size_hint: None, None
        size: dp(72), dp(72)
        radius: [dp(36)]
        elevation: 0
        pos_hint: {"center_x": 0.5}
        md_bg_color: app.c_accent_soft

        MDAnchorLayout:
            MDIcon:
                size_hint: None, None
                size: self.font_size + dp(8), self.font_size + dp(8)
                icon: root.icon
                font_size: "34sp"
                halign: "center"
                valign: "middle"
                theme_text_color: "Custom"
                text_color: app.c_accent

    MDLabel:
        text: root.headline
        font_name: "RobotoLight"
        font_size: "20sp"
        halign: "center"
        adaptive_height: True
        theme_text_color: "Custom"
        text_color: app.c_text

    MDLabel:
        text: root.body
        font_style: "Body2"
        halign: "center"
        adaptive_height: True
        theme_text_color: "Custom"
        text_color: app.c_muted

    MDLabel:
        text: root.module_tag
        font_style: "Caption"
        bold: True
        halign: "center"
        adaptive_height: True
        theme_text_color: "Custom"
        text_color: app.c_accent


<ZonePage>:
    md_bg_color: app.c_bg

    MDScrollView:
        do_scroll_x: False
        bar_width: 0

        MDBoxLayout:
            orientation: "vertical"
            padding: dp(22), dp(10), dp(22), dp(112)
            spacing: dp(16)
            adaptive_height: True

            ZoneCard:
                icon: root.icon
                headline: root.headline
                body: root.body
                module_tag: root.module_tag


<CutoutSettingsCard>:
    orientation: "vertical"
    size_hint_y: None
    height: self.minimum_height
    radius: [dp(24)]
    elevation: 0
    padding: dp(20)
    spacing: dp(12)
    md_bg_color: app.c_surface
    line_color: app.c_border
    line_width: dp(0.8)

    MDLabel:
        text: "Background removal"
        font_name: "RobotoLight"
        font_size: "20sp"
        adaptive_height: True
        theme_text_color: "Custom"
        text_color: app.c_text

    MDLabel:
        text: "On-device is private and free. Cloud services upload each photo and need your own API key; any failure falls back to on-device."
        font_style: "Caption"
        adaptive_height: True
        theme_text_color: "Custom"
        text_color: app.c_muted

    StackLayout:
        id: provider_chips
        size_hint_y: None
        height: self.minimum_height
        spacing: dp(8)

    Collapsible:
        shown: bool(root.field_hint)

        MDTextField:
            id: cred_field
            hint_text: root.field_hint
            password: root.field_secret

    MDLabel:
        text: root.privacy_note
        font_style: "Caption"
        bold: True
        adaptive_height: True
        theme_text_color: "Custom"
        text_color: app.c_accent

    PillButton:
        icon: "content-save-outline"
        text: "Save engine"
        filled: True
        on_release: root.save()


# ---------------------------------------------------------------- stylist

<ColorDot>:
    size_hint: None, None
    size: dp(22), dp(22)
    canvas:
        Color:
            rgba: app.c_surface
        Ellipse:
            pos: self.x - dp(2), self.y - dp(2)
            size: self.width + dp(4), self.height + dp(4)
        Color:
            rgba: root.rgba
        Ellipse:
            pos: self.pos
            size: self.size


<ScoreRing>:
    size_hint: None, None
    size: dp(58), dp(58)
    canvas:
        Color:
            rgba: app.c_border
        Line:
            circle: (self.width / 2.0, self.height / 2.0, self.width / 2.0 - dp(4))
            width: dp(3)
        Color:
            rgba: app.c_accent
        Line:
            circle: (self.width / 2.0, self.height / 2.0, self.width / 2.0 - dp(4), 0, 360 * max(0.01, min(1.0, root.value)))
            width: dp(3)
            cap: "round"

    MDLabel:
        text: root.label
        font_style: "Subtitle2"
        bold: True
        halign: "center"
        valign: "middle"
        theme_text_color: "Custom"
        text_color: app.c_text


<ItemThumb>:
    orientation: "vertical"
    size_hint: None, None
    size: dp(98), dp(124)
    radius: [dp(18)]
    elevation: 0
    padding: dp(6)
    spacing: dp(2)
    md_bg_color: app.c_frame
    line_color: app.c_border
    line_width: dp(0.8)

    MDRelativeLayout:

        MDCard:
            size_hint: 0.62, 0.62
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            radius: [dp(12)]
            elevation: 0
            opacity: 0 if root.source else 1
            md_bg_color: get_color_from_hex(root.swatch) if root.swatch else app.c_border

        Image:
            source: root.source
            fit_mode: "contain"
            nocache: True
            opacity: 1 if root.source else 0

    MDLabel:
        text: root.caption
        font_style: "Caption"
        bold: True
        halign: "center"
        shorten: True
        size_hint_y: None
        height: dp(16)
        theme_text_color: "Custom"
        text_color: app.c_text

    MDLabel:
        text: root.layer_text
        font_style: "Overline"
        halign: "center"
        size_hint_y: None
        height: dp(14)
        theme_text_color: "Custom"
        text_color: app.c_accent


<ReasonPill>:
    size_hint: None, None
    height: dp(26)
    width: reason_label.width + dp(22)
    padding: dp(11), 0
    radius_v: dp(13)
    bg: 0, 0, 0, 0
    line: app.c_border

    TxtW:
        id: reason_label
        text: root.text
        font_size: "11sp"
        color: app.c_muted
        pos_hint: {"center_y": 0.5}


<ScoreMeter>:
    size_hint: None, None
    size: dp(76), dp(76)

    canvas.before:
        Color:
            rgba: app.c_surface
        Ellipse:
            pos: 0, 0
            size: self.size
        Color:
            rgba: app.c_border
        Line:
            circle: (self.width / 2.0, self.height / 2.0, self.width / 2.0 - dp(6))
            width: dp(1)
        Color:
            rgba: app.c_accent
        Line:
            circle: (self.width / 2.0, self.height / 2.0, self.width / 2.0 - dp(6), 0, 360 * max(0.01, min(1.0, root.value)))
            width: dp(2.2)
            cap: "round"

    Label:
        text: root.label
        font_name: "RobotoLight"
        font_size: "21sp"
        color: app.c_text
        size_hint: 1, None
        height: dp(26)
        pos_hint: {"center_x": 0.5, "center_y": 0.58}

    Label:
        text: root.rule
        font_name: "RobotoMedium"
        font_size: "7sp"
        color: app.c_muted
        size_hint: 1, None
        height: dp(12)
        pos_hint: {"center_x": 0.5, "center_y": 0.33}


<InteractionChip>:
    size_hint: None, None
    height: dp(44)
    width: ic_row.minimum_width + dp(34)
    padding: dp(17), 0
    radius_v: dp(22)
    bg: app.c_accent if root.active else (0, 0, 0, 0)
    line: app.c_accent if root.active else app.c_text
    line_w: dp(0.8) + dp(2.4) * root.pulse
    opacity: 0.75 if self.state == "down" else 1

    BoxLayout:
        id: ic_row
        size_hint_x: None
        width: self.minimum_width
        spacing: dp(8)

        Ico:
            icon: root.icon
            font_size: "19sp"
            color: app.c_on_accent if root.active else app.c_text
            pos_hint: {"center_y": 0.5}

        TxtW:
            text: root.text
            font_name: "RobotoMedium"
            font_size: "11sp"
            color: app.c_on_accent if root.active else app.c_text
            pos_hint: {"center_y": 0.5}


<OutfitCard>:
    orientation: "vertical"
    padding: dp(14), dp(14), dp(16), dp(14)
    spacing: dp(8)
    radius_v: dp(32)
    bg: app.c_surface
    line: app.c_laser if root.shuffle else app.c_border
    line_w: dp(1.4) if root.shuffle else dp(0.8)

    # ---- floating flat-lay stage
    RelativeLayout:
        size_hint_y: None
        height: dp(340)

        FlatLay:
            id: flatlay

        TxtW:
            text: root.kicker
            font_name: "RobotoMedium"
            font_size: "10sp"
            color: app.c_laser if root.shuffle else app.c_muted
            pos_hint: {"x": 0.03, "top": 0.985}

        ScoreMeter:
            value: root.score
            label: root.score_text
            rule: root.rule_short
            pos_hint: {"right": 1, "top": 1}

    BoxLayout:
        orientation: "vertical"
        size_hint_y: None
        height: self.minimum_height
        spacing: dp(2)
        padding: dp(4), 0

        Txt:
            text: root.title
            font_name: "RobotoLight"
            font_size: "21sp"

        Txt:
            text: root.subtitle
            font_size: "12sp"
            color: app.c_muted
            shorten: True
            shorten_from: "right"
            max_lines: 1

    StackLayout:
        id: reasons
        size_hint_y: None
        height: self.minimum_height
        spacing: dp(6)
        padding: dp(4), 0

    Widget:

    BoxLayout:
        size_hint_y: None
        height: dp(46)
        padding: dp(2), 0
        spacing: dp(8)

        InteractionChip:
            icon: "check-circle" if root.worn else "check-circle-outline"
            text: "WORN TODAY" if root.worn else "WEAR TODAY"
            active: root.worn
            pos_hint: {"center_y": 0.5}
            on_release: root.dispatch("on_wear")

        Widget:

        IconTap:
            icon: "heart" if root.saved else "heart-outline"
            font_size: root.heart_size
            size_hint: None, None
            size: dp(44), dp(44)
            pos_hint: {"center_y": 0.5}
            color: get_color_from_hex("#FF2D55") if root.saved else app.c_text
            on_release: root.dispatch("on_save")


<ListRow>:
    orientation: "horizontal"
    size_hint_y: None
    height: dp(64)
    padding: dp(14), dp(8), dp(6), dp(8)
    spacing: dp(12)
    radius_v: dp(18)
    bg: app.c_surface
    line: app.c_border

    BoxLayout:
        id: dots
        size_hint: None, None
        width: self.minimum_width
        height: dp(22)
        spacing: -dp(6)
        pos_hint: {"center_y": 0.5}

    BoxLayout:
        orientation: "vertical"
        size_hint_y: None
        height: self.minimum_height
        spacing: dp(2)
        pos_hint: {"center_y": 0.5}

        Txt:
            text: root.title
            font_name: "RobotoMedium"
            font_size: "13sp"
            shorten: True
            shorten_from: "right"
            max_lines: 1

        Txt:
            text: root.subtitle
            font_size: "11sp"
            color: app.c_muted
            shorten: True
            shorten_from: "right"
            max_lines: 1

    Surface:
        size_hint: None, None
        height: dp(24)
        width: (badge_label.width + dp(18)) if root.badge else 0
        padding: dp(9), 0
        radius_v: dp(12)
        opacity: 1 if root.badge else 0
        bg: app.c_accent if root.badge_on else (0, 0, 0, 0)
        line: app.c_accent if root.badge else (0, 0, 0, 0)
        pos_hint: {"center_y": 0.5}

        TxtW:
            id: badge_label
            text: root.badge
            font_name: "RobotoMedium"
            font_size: "9sp"
            color: app.c_on_accent if root.badge_on else app.c_accent
            pos_hint: {"center_y": 0.5}

    IconTap:
        icon: root.action_icon or "blank"
        font_size: "20sp"
        size_hint: None, None
        size: (dp(40), dp(40)) if root.action_icon else (0, 0)
        opacity: 1 if root.action_icon else 0
        color: app.c_muted
        pos_hint: {"center_y": 0.5}
        on_release: root.dispatch("on_action") if root.action_icon else None


<LookCard>:
    orientation: "vertical"
    size_hint: None, None
    size: dp(248), dp(326)
    padding: dp(12)
    spacing: dp(8)
    radius_v: dp(26)
    bg: app.c_surface
    line: app.c_border
    opacity: 0.85 if self.state == "down" else 1

    LookCollage:
        id: collage
        size_hint_y: None
        height: dp(178)

    BoxLayout:
        id: dots
        size_hint: None, None
        width: self.minimum_width
        height: dp(14)
        spacing: -dp(4)
        padding: dp(2), 0

    Label:
        text: root.title
        font_name: "RobotoMedium"
        font_size: "13sp"
        color: app.c_text
        max_lines: 2
        shorten: True
        shorten_from: "right"
        size_hint_y: None
        height: dp(36)
        halign: "left"
        valign: "top"
        text_size: self.width, self.height

    Txt:
        text: root.subtitle
        font_size: "11sp"
        color: app.c_muted
        shorten: True
        shorten_from: "right"
        max_lines: 1

    Surface:
        size_hint_y: None
        height: dp(32)
        radius_v: dp(16)
        bg: app.c_accent if root.cta_active else (0, 0, 0, 0)
        line: app.c_accent if root.cta_active else app.c_text
        line_w: dp(0.8) + dp(2.2) * root.pulse
        on_release: root.dispatch("on_chip")

        Label:
            text: root.cta
            font_name: "RobotoMedium"
            font_size: "9sp"
            color: app.c_on_accent if root.cta_active else app.c_text


# Soft layered spotlight behind floating garments (RelativeLayout canvas ->
# local coordinates). Dark: faint white glow; light: gentle grey wash.
<FlatLay,LookCollage>:
    spot_alpha: 0.05 if app.theme_cls.theme_style == "Dark" else 0.55
    canvas.before:
        Color:
            rgba: (1, 1, 1, root.spot_alpha * 0.45) if app.theme_cls.theme_style == "Dark" else (app.c_surface_alt[0], app.c_surface_alt[1], app.c_surface_alt[2], root.spot_alpha * 0.6)
        Ellipse:
            pos: self.width / 2.0 - min(self.width, self.height) * 0.48, self.height * 0.48 - min(self.width, self.height) * 0.48
            size: min(self.width, self.height) * 0.96, min(self.width, self.height) * 0.96
        Color:
            rgba: (1, 1, 1, root.spot_alpha * 0.7) if app.theme_cls.theme_style == "Dark" else (app.c_surface_alt[0], app.c_surface_alt[1], app.c_surface_alt[2], root.spot_alpha * 0.85)
        Ellipse:
            pos: self.width / 2.0 - min(self.width, self.height) * 0.38, self.height * 0.48 - min(self.width, self.height) * 0.38
            size: min(self.width, self.height) * 0.76, min(self.width, self.height) * 0.76
        Color:
            rgba: (1, 1, 1, root.spot_alpha) if app.theme_cls.theme_style == "Dark" else (app.c_surface_alt[0], app.c_surface_alt[1], app.c_surface_alt[2], root.spot_alpha * 1.25)
        Ellipse:
            pos: self.width / 2.0 - min(self.width, self.height) * 0.28, self.height * 0.48 - min(self.width, self.height) * 0.28
            size: min(self.width, self.height) * 0.56, min(self.width, self.height) * 0.56


<GarmentTile>:
    padding: dp(8)
    radius_v: dp(22)
    bg: get_color_from_hex(root.hex_color)
    line: app.c_border

    Label:
        text: root.caption
        font_name: "RobotoMedium"
        font_size: "8sp"
        color: root.ink
        valign: "bottom"
        text_size: self.size
        halign: "center"


<MissingFrame>:
    canvas.before:
        Color:
            rgba: app.c_muted
        Line:
            rounded_rectangle: (dp(1), dp(1), self.width - dp(2), self.height - dp(2), dp(18))
            dash_length: dp(5)
            dash_offset: dp(4)
            width: 1
        Color:
            rgba: get_color_from_hex(root.hex_color)
        Ellipse:
            pos: self.width / 2.0 - dp(9), self.height * 0.22 - dp(9)
            size: dp(18), dp(18)

    Ico:
        icon: "plus"
        font_size: "26sp"
        color: app.c_muted
        pos_hint: {"center_x": 0.5, "center_y": 0.56}

    Label:
        text: root.caption
        font_name: "RobotoMedium"
        font_size: "8sp"
        color: app.c_muted
        size_hint: 1, None
        height: dp(12)
        pos_hint: {"center_x": 0.5, "y": 0.04}


<LookDetail>:
    size_hint: None, None
    width: min(Window.width - dp(32), dp(400))
    height: detail_box.height
    background_color: 0, 0, 0, 0
    overlay_color: 0, 0, 0, 0.55

    MDCard:
        id: detail_box
        orientation: "vertical"
        size_hint: 1, None
        height: self.minimum_height
        radius: [dp(28)]
        elevation: 0
        padding: dp(20)
        spacing: dp(12)
        md_bg_color: app.c_surface
        line_color: app.c_border
        line_width: dp(0.8)

        LuxKicker:
            text: root.kicker
            text_color: app.c_accent

        LuxTitle:
            text: root.title

        LuxBody:
            text: root.subtitle

        BoxLayout:
            id: rows
            orientation: "vertical"
            size_hint_y: None
            height: self.minimum_height
            spacing: dp(8)

        MDBoxLayout:
            size_hint_y: None
            height: dp(50)
            spacing: dp(10)

            PillButton:
                icon: root.primary_icon
                text: root.primary_text
                filled: True
                on_release: root.dispatch("on_primary")

            PillButton:
                icon: "close"
                text: "Close"
                filled: False
                on_release: root.dismiss()


<SectionCard@MDCard>:
    orientation: "vertical"
    size_hint_y: None
    height: self.minimum_height
    radius: [dp(24)]
    elevation: 0
    padding: dp(18)
    spacing: dp(10)
    md_bg_color: app.c_surface
    line_color: app.c_border
    line_width: dp(0.8)


<CardTitle@MDLabel>:
    font_name: "RobotoLight"
    font_size: "20sp"
    adaptive_height: True
    theme_text_color: "Custom"
    text_color: app.c_text


<CardCaption@MDLabel>:
    font_style: "Caption"
    adaptive_height: True
    theme_text_color: "Custom"
    text_color: app.c_muted


<StylistZone>:
    md_bg_color: app.c_bg

    MDScrollView:
        do_scroll_x: False
        bar_width: 0

        MDBoxLayout:
            orientation: "vertical"
            adaptive_height: True
            padding: dp(22), dp(10), dp(22), dp(112)
            spacing: dp(16)

            MDBoxLayout:
                orientation: "vertical"
                adaptive_height: True
                spacing: dp(4)

                LuxKicker:
                    text: "AI STYLIST"
                    text_color: app.c_accent

                LuxDisplay:
                    text: root.headline

                MDLabel:
                    text: root.subline
                    font_style: "Body2"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_muted

            MDCard:
                size_hint: None, None
                height: dp(36)
                width: weather_row.minimum_width + dp(28)
                radius: [dp(18)]
                elevation: 0
                padding: dp(12), 0, dp(16), 0
                md_bg_color: app.c_surface
                line_color: app.c_accent if root.weather_live else app.c_border
                line_width: dp(0.8)
                on_release: root.refresh_weather()

                MDBoxLayout:
                    id: weather_row
                    adaptive_width: True
                    spacing: dp(8)

                    MDIcon:
                        icon: root.weather_icon
                        font_size: "20sp"
                        size_hint: None, None
                        size: dp(22), dp(22)
                        pos_hint: {"center_y": 0.5}
                        theme_text_color: "Custom"
                        text_color: app.c_accent

                    MDLabel:
                        text: root.weather_text
                        font_style: "Body2"
                        bold: True
                        adaptive_width: True
                        pos_hint: {"center_y": 0.5}
                        theme_text_color: "Custom"
                        text_color: app.c_text

            StackLayout:
                id: occasion_chips
                size_hint_y: None
                height: self.minimum_height
                spacing: dp(8)

            MDRelativeLayout:
                size_hint_y: None
                height: dp(596)

                Carousel:
                    id: deck
                    direction: "right"
                    loop: False
                    on_index: root.on_deck_index()

                MDCard:
                    orientation: "vertical"
                    size_hint: 1, None
                    height: self.minimum_height
                    pos_hint: {"center_y": 0.5}
                    radius: [dp(28)]
                    elevation: 0
                    padding: dp(24)
                    spacing: dp(12)
                    opacity: 0 if root.has_outfits else 1
                    md_bg_color: app.c_surface
                    line_color: app.c_border
                    line_width: dp(0.8)

                    MDCard:
                        size_hint: None, None
                        size: dp(72), dp(72)
                        radius: [dp(36)]
                        elevation: 0
                        pos_hint: {"center_x": 0.5}
                        md_bg_color: app.c_accent_soft

                        MDAnchorLayout:
                            MDIcon:
                                icon: "creation"
                                font_size: "34sp"
                                size_hint: None, None
                                size: self.font_size + dp(8), self.font_size + dp(8)
                                halign: "center"
                                valign: "middle"
                                theme_text_color: "Custom"
                                text_color: app.c_accent

                    MDLabel:
                        text: root.empty_title
                        font_name: "RobotoLight"
                        font_size: "20sp"
                        halign: "center"
                        adaptive_height: True
                        theme_text_color: "Custom"
                        text_color: app.c_text

                    MDLabel:
                        text: root.empty_body
                        font_style: "Body2"
                        halign: "center"
                        adaptive_height: True
                        theme_text_color: "Custom"
                        text_color: app.c_muted

            MDBoxLayout:
                size_hint_y: None
                height: dp(40)
                opacity: 1 if root.has_outfits else 0

                MDIconButton:
                    icon: "chevron-left"
                    pos_hint: {"center_y": 0.5}
                    theme_icon_color: "Custom"
                    icon_color: app.c_text
                    on_release: root.ids.deck.load_previous()

                Widget:

                BoxLayout:
                    id: dots
                    size_hint: None, None
                    width: self.minimum_width
                    height: dp(8)
                    spacing: dp(6)
                    pos_hint: {"center_y": 0.5}

                Widget:

                MDIconButton:
                    icon: "chevron-right"
                    pos_hint: {"center_y": 0.5}
                    theme_icon_color: "Custom"
                    icon_color: app.c_text
                    on_release: root.ids.deck.load_next()

            MDBoxLayout:
                size_hint_y: None
                height: dp(52)
                spacing: dp(12)

                PillButton:
                    icon: "shuffle-variant"
                    text: "Virtual shuffle"
                    filled: True
                    on_release: root.shuffle()

                PillButton:
                    icon: "refresh"
                    text: "Refresh"
                    filled: False
                    on_release: root.generate()

            # ---------- trend discovery (swipeable lookbook stream)
            MDBoxLayout:
                orientation: "vertical"
                adaptive_height: True
                spacing: dp(10)
                padding: 0, dp(12), 0, 0

                MDBoxLayout:
                    size_hint_y: None
                    height: dp(52)

                    MDBoxLayout:
                        orientation: "vertical"
                        adaptive_height: True
                        spacing: dp(2)
                        pos_hint: {"center_y": 0.5}

                        LuxKicker:
                            text: "TREND DISCOVERY"

                        LuxTitle:
                            text: "Looks to recreate"

                    MDIconButton:
                        icon: "sync"
                        pos_hint: {"center_y": 0.5}
                        theme_icon_color: "Custom"
                        icon_color: app.c_text
                        on_release: root.sync_trends_now()

                LuxBody:
                    text: root.trend_caption

                ScrollView:
                    id: trend_scroll
                    size_hint_y: None
                    height: dp(330) if root.has_trends else 0
                    opacity: 1 if root.has_trends else 0
                    do_scroll_y: False
                    bar_width: 0

                    BoxLayout:
                        id: trend_list
                        size_hint_x: None
                        width: self.minimum_width
                        spacing: dp(12)
                        padding: 0, 0, dp(8), 0

            # ---------- smart wishlist / wardrobe gaps
            MDBoxLayout:
                orientation: "vertical"
                adaptive_height: True
                spacing: dp(10)
                padding: 0, dp(8), 0, 0

                LuxKicker:
                    text: "SMART WISHLIST"

                LuxTitle:
                    text: "Wardrobe gaps"

                LuxBody:
                    text: root.gap_caption

                BoxLayout:
                    id: gap_list
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

            # ---------- capsule packing wizard
            SectionCard:

                CardTitle:
                    text: "Capsule packing wizard"

                CardCaption:
                    text: "Fewest pieces, most outfits - planned day by day for the forecast."

                MDTextField:
                    id: pack_dest
                    hint_text: "Destination city"

                MDTextField:
                    id: pack_start
                    hint_text: "Start date (YYYY-MM-DD)"

                MDBoxLayout:
                    size_hint_y: None
                    height: dp(48)
                    spacing: dp(4)

                    MDLabel:
                        text: "Trip length"
                        font_style: "Body2"
                        bold: True
                        pos_hint: {"center_y": 0.5}
                        theme_text_color: "Custom"
                        text_color: app.c_text

                    MDIconButton:
                        icon: "minus"
                        pos_hint: {"center_y": 0.5}
                        theme_icon_color: "Custom"
                        icon_color: app.c_accent
                        on_release: root.change_days(-1)

                    MDLabel:
                        text: root.pack_days_text
                        font_style: "Subtitle1"
                        bold: True
                        halign: "center"
                        size_hint_x: None
                        width: dp(72)
                        pos_hint: {"center_y": 0.5}
                        theme_text_color: "Custom"
                        text_color: app.c_text

                    MDIconButton:
                        icon: "plus"
                        pos_hint: {"center_y": 0.5}
                        theme_icon_color: "Custom"
                        icon_color: app.c_accent
                        on_release: root.change_days(1)

                StackLayout:
                    id: pack_occasions
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

                PillButton:
                    icon: "bag-suitcase-outline"
                    text: "Build capsule"
                    filled: True
                    on_release: root.build_capsule()

                CardCaption:
                    text: root.pack_summary

                StackLayout:
                    id: pack_items
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

                BoxLayout:
                    id: pack_plan
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

            # ---------- live data (opt-in)
            SectionCard:

                CardTitle:
                    text: "Live data (opt-in)"

                CardCaption:
                    text: "Off by default. Weather sends only your city's rounded coordinates to Open-Meteo; trend sync downloads only the feeds you add. Nothing about you or your wardrobe is uploaded."

                MDBoxLayout:
                    size_hint_y: None
                    height: dp(40)

                    MDLabel:
                        text: "Weather (Open-Meteo)"
                        font_style: "Body2"
                        bold: True
                        pos_hint: {"center_y": 0.5}
                        theme_text_color: "Custom"
                        text_color: app.c_text

                    ChipButton:
                        text: "On" if root.weather_on else "Off"
                        icon: "check" if root.weather_on else "close"
                        selected: root.weather_on
                        pos_hint: {"center_y": 0.5}
                        on_release: root.toggle_weather()

                MDTextField:
                    id: city_field
                    hint_text: "Your city (for local weather)"

                PillButton:
                    icon: "map-marker-outline"
                    text: "Set city"
                    filled: False
                    on_release: root.set_city()

                MDBoxLayout:
                    size_hint_y: None
                    height: dp(40)

                    MDLabel:
                        text: "Trend feeds (RSS / Atom)"
                        font_style: "Body2"
                        bold: True
                        pos_hint: {"center_y": 0.5}
                        theme_text_color: "Custom"
                        text_color: app.c_text

                    ChipButton:
                        text: "On" if root.trends_on else "Off"
                        icon: "check" if root.trends_on else "close"
                        selected: root.trends_on
                        pos_hint: {"center_y": 0.5}
                        on_release: root.toggle_trends()

                MDTextField:
                    id: feed_field
                    hint_text: "Feed URL (https://...)"

                PillButton:
                    icon: "rss"
                    text: "Add feed"
                    filled: False
                    on_release: root.add_feed()

                BoxLayout:
                    id: feed_list
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

    Toast:
        id: toast


<MetricTile>:
    orientation: "vertical"
    radius: [dp(20)]
    elevation: 0
    padding: dp(14), dp(12)
    spacing: dp(2)
    md_bg_color: app.c_surface
    line_color: app.c_border
    line_width: dp(0.8)

    MDBoxLayout:
        size_hint_y: None
        height: dp(28)
        spacing: dp(8)

        MDCard:
            size_hint: None, None
            size: dp(28), dp(28)
            radius: [dp(14)]
            elevation: 0
            pos_hint: {"center_y": 0.5}
            md_bg_color: get_color_from_hex(root.swatch) if root.swatch else app.c_accent_soft
            line_color: app.c_border if root.swatch else (0, 0, 0, 0)
            line_width: dp(0.8)

            MDAnchorLayout:
                MDIcon:
                    size_hint: None, None
                    size: self.font_size + dp(8), self.font_size + dp(8)
                    icon: root.icon
                    font_size: "16sp"
                    halign: "center"
                    valign: "middle"
                    opacity: 0 if root.swatch else 1
                    theme_text_color: "Custom"
                    text_color: app.c_accent

        MDLabel:
            text: root.label
            font_style: "Overline"
            bold: True
            pos_hint: {"center_y": 0.5}
            theme_text_color: "Custom"
            text_color: app.c_muted

    Widget:
        size_hint_y: None
        height: dp(6)

    MDLabel:
        text: root.value
        font_name: "RobotoLight"
        font_size: "20sp"
        adaptive_height: True
        theme_text_color: "Custom"
        text_color: app.c_text if root.value != "—" else app.c_muted

    MDLabel:
        text: root.caption
        font_style: "Caption"
        adaptive_height: True
        theme_text_color: "Custom"
        text_color: app.c_muted

    Widget:


<LegendDot@MDBoxLayout>:
    dot_color: 1, 1, 1, 1
    text: ""
    adaptive_width: True
    spacing: dp(6)

    MDIcon:
        icon: "circle"
        font_size: "12sp"
        size_hint: None, None
        size: dp(12), dp(12)
        pos_hint: {"center_y": 0.5}
        theme_text_color: "Custom"
        text_color: root.dot_color

    MDLabel:
        text: root.text
        font_style: "Caption"
        adaptive_width: True
        pos_hint: {"center_y": 0.5}
        theme_text_color: "Custom"
        text_color: app.c_muted


<BodyPhotoFrame>:
    size_hint_y: None
    height: self.width * 1.2
    radius: [dp(28)]
    elevation: 0
    padding: 0
    md_bg_color: app.c_frame
    line_color: app.c_accent if root.manual_mode else app.c_border
    line_width: dp(2) if root.manual_mode else dp(1.2)

    MDRelativeLayout:

        MDBoxLayout:
            orientation: "vertical"
            adaptive_height: True
            spacing: dp(12)
            padding: dp(28), 0
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            opacity: 0 if root.photo_source else 1

            MDCard:
                size_hint: None, None
                size: dp(88), dp(88)
                radius: [dp(44)]
                elevation: 0
                pos_hint: {"center_x": 0.5}
                md_bg_color: app.c_accent_soft

                MDAnchorLayout:
                    MDIcon:
                        size_hint: None, None
                        size: self.font_size + dp(8), self.font_size + dp(8)
                        icon: "human-handsup"
                        font_size: "44sp"
                        halign: "center"
                        valign: "middle"
                        theme_text_color: "Custom"
                        text_color: app.c_accent

            MDLabel:
                text: "Add a full-body photo"
                font_style: "Subtitle1"
                bold: True
                halign: "center"
                adaptive_height: True
                theme_text_color: "Custom"
                text_color: app.c_text

            MDLabel:
                text: "Stand straight facing the camera, arms slightly away from your sides, fitted clothes, plain background, daylight."
                font_style: "Caption"
                halign: "center"
                adaptive_height: True
                theme_text_color: "Custom"
                text_color: app.c_muted

        Image:
            id: photo
            source: root.photo_source
            fit_mode: "contain"
            nocache: True
            size_hint: 0.94, 0.94
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            opacity: 1 if root.photo_source else 0

        LandmarkOverlay:
            frame: root
            image: photo
            points: root.points
            face_box: root.face_box
            skin_point: root.skin_point
            manual_mode: root.manual_mode

        MDCard:
            size_hint: None, None
            height: dp(28)
            width: method_row.minimum_width + dp(22)
            radius: [dp(14)]
            elevation: 0
            padding: dp(10), 0, dp(12), 0
            pos_hint: {"x": 0.05, "top": 0.96}
            opacity: 1 if root.photo_source and root.pill_text else 0
            md_bg_color: app.c_surface
            line_color: app.c_border
            line_width: dp(0.8)

            MDBoxLayout:
                id: method_row
                adaptive_width: True
                spacing: dp(4)

                MDIcon:
                    icon: "gesture-tap" if root.manual_mode else "shield-lock-outline"
                    font_size: "14sp"
                    size_hint: None, None
                    size: dp(16), dp(16)
                    pos_hint: {"center_y": 0.5}
                    theme_text_color: "Custom"
                    text_color: app.c_accent

                MDLabel:
                    text: root.pill_text
                    font_style: "Caption"
                    bold: True
                    adaptive_width: True
                    pos_hint: {"center_y": 0.5}
                    theme_text_color: "Custom"
                    text_color: app.c_text


<ProfileZone>:
    md_bg_color: app.c_bg

    MDScrollView:
        do_scroll_x: False
        bar_width: 0

        MDBoxLayout:
            orientation: "vertical"
            padding: dp(22), dp(10), dp(22), dp(112)
            spacing: dp(16)
            adaptive_height: True

            MDBoxLayout:
                orientation: "vertical"
                adaptive_height: True
                spacing: dp(4)

                LuxKicker:
                    text: "BIOMETRIC PROFILE"
                    text_color: app.c_accent

                LuxDisplay:
                    text: "Body & Style"

                MDLabel:
                    text: root.status_text
                    font_style: "Body2"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_muted

            BodyPhotoFrame:
                id: body_frame
                photo_source: root.photo_source
                points: root.points
                face_box: root.face_box
                skin_point: root.skin_point
                manual_mode: root.manual_mode
                pill_text: root.method_pill
                on_point_tapped: root.add_manual_point(args[1], args[2])

            MDBoxLayout:
                size_hint_y: None
                height: dp(18) if (root.points or root.manual_mode) else 0
                opacity: 1 if (root.points or root.manual_mode) else 0
                spacing: dp(16)
                padding: dp(4), 0

                LegendDot:
                    dot_color: app.c_accent
                    text: "Shoulders"

                LegendDot:
                    dot_color: get_color_from_hex("#F59E0B")
                    text: "Waist"

                LegendDot:
                    dot_color: get_color_from_hex("#EC4899")
                    text: "Hips"

                LegendDot:
                    dot_color: app.c_text
                    text: "Skin sample"

            # ---------- manual tap override banner
            Collapsible:
                shown: root.manual_mode

                MDCard:
                    orientation: "vertical"
                    size_hint_y: None
                    height: self.minimum_height
                    radius: [dp(20)]
                    elevation: 0
                    padding: dp(16)
                    spacing: dp(6)
                    md_bg_color: app.c_accent_soft
                    line_color: app.c_accent
                    line_width: dp(0.8)

                    LuxKicker:
                        text: root.manual_step_text
                        text_color: app.c_accent

                    MDLabel:
                        text: root.manual_prompt
                        font_style: "Subtitle1"
                        bold: True
                        adaptive_height: True
                        theme_text_color: "Custom"
                        text_color: app.c_text

                    MDLabel:
                        text: "Tap the outer edge of your body at that height. Left and right are as seen in the photo."
                        font_style: "Caption"
                        adaptive_height: True
                        theme_text_color: "Custom"
                        text_color: app.c_muted

                    Widget:
                        size_hint_y: None
                        height: dp(4)

                    MDBoxLayout:
                        size_hint_y: None
                        height: dp(52)
                        spacing: dp(12)

                        PillButton:
                            icon: "undo-variant"
                            text: "Undo"
                            filled: False
                            on_release: root.undo_manual_point()

                        PillButton:
                            icon: "close"
                            text: "Cancel"
                            filled: False
                            on_release: root.cancel_manual()

            # ---------- capture actions
            Collapsible:
                shown: not root.manual_mode
                spacing: dp(12)

                MDBoxLayout:
                    size_hint_y: None
                    height: dp(52)
                    spacing: dp(12)

                    PillButton:
                        icon: "camera-outline"
                        text: "Camera"
                        filled: True
                        on_release: root.open_camera()

                    PillButton:
                        icon: "image-plus"
                        text: "Upload photo"
                        filled: False
                        on_release: root.open_gallery()

                Collapsible:
                    shown: root.has_photo

                    MDBoxLayout:
                        size_hint_y: None
                        height: dp(52)
                        spacing: dp(12)

                        PillButton:
                            icon: "gesture-tap"
                            text: "Adjust points"
                            filled: False
                            on_release: root.start_manual()

                        PillButton:
                            icon: "delete-outline"
                            text: "Delete photo"
                            filled: False
                            on_release: root.delete_photo()

            SectionTitle:
                text: "YOUR MEASUREMENTS"

            GridLayout:
                cols: 2
                spacing: dp(12)
                size_hint_y: None
                height: self.minimum_height
                row_default_height: dp(122)
                row_force_default: True

                MetricTile:
                    icon: "human-handsup"
                    label: "BODY SHAPE"
                    value: root.shape_value
                    caption: root.shape_caption

                MetricTile:
                    icon: "arrow-expand-horizontal"
                    label: "SHOULDER : WAIST"
                    value: root.sw_value
                    caption: root.sw_caption

                MetricTile:
                    icon: "tape-measure"
                    label: "WAIST : HIP"
                    value: root.wh_value
                    caption: root.wh_caption

                MetricTile:
                    icon: "water-outline"
                    label: "UNDERTONE"
                    value: root.tone_value
                    caption: root.tone_caption
                    swatch: root.tone_swatch

                MetricTile:
                    icon: "brightness-6"
                    label: "SKIN DEPTH"
                    value: root.depth_value
                    caption: root.depth_caption

                MetricTile:
                    icon: "palette-outline"
                    label: "COLOUR SEASON"
                    value: root.season_value
                    caption: root.season_caption

            # ---------- palette
            MDCard:
                orientation: "vertical"
                size_hint_y: None
                height: self.minimum_height
                radius: [dp(24)]
                elevation: 0
                padding: dp(20)
                spacing: dp(10)
                md_bg_color: app.c_surface
                line_color: app.c_border
                line_width: dp(0.8)

                MDLabel:
                    text: "Your colour palette"
                    font_name: "RobotoLight"
                    font_size: "20sp"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_text

                MDLabel:
                    text: root.palette_caption
                    font_style: "Caption"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_muted

                SectionTitle:
                    text: "BEST COLOURS"

                StackLayout:
                    id: best_colors
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

                SectionTitle:
                    text: "BEST NEUTRALS"

                StackLayout:
                    id: best_neutrals
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

                SectionTitle:
                    text: "GO EASY ON"

                StackLayout:
                    id: avoid_colors
                    size_hint_y: None
                    height: self.minimum_height
                    spacing: dp(8)

            # ---------- privacy note
            MDCard:
                orientation: "horizontal"
                size_hint_y: None
                height: self.minimum_height
                radius: [dp(20)]
                elevation: 0
                padding: dp(14)
                spacing: dp(12)
                md_bg_color: app.c_surface
                line_color: app.c_border
                line_width: dp(0.8)

                MDCard:
                    size_hint: None, None
                    size: dp(40), dp(40)
                    radius: [dp(20)]
                    elevation: 0
                    pos_hint: {"center_y": 0.5}
                    md_bg_color: app.c_accent_soft

                    MDAnchorLayout:
                        MDIcon:
                            size_hint: None, None
                            size: self.font_size + dp(8), self.font_size + dp(8)
                            icon: "shield-lock-outline"
                            font_size: "22sp"
                            halign: "center"
                            valign: "middle"
                            theme_text_color: "Custom"
                            text_color: app.c_accent

                MDLabel:
                    text: "Body photos are analysed on this device only and are never uploaded, even when a cloud cut-out service is switched on."
                    font_style: "Caption"
                    adaptive_height: True
                    pos_hint: {"center_y": 0.5}
                    theme_text_color: "Custom"
                    text_color: app.c_muted

            CutoutSettingsCard:
                id: cutout_card

            MDCard:
                orientation: "vertical"
                padding: dp(20)
                spacing: dp(10)
                size_hint_y: None
                height: self.minimum_height
                radius: [dp(24)]
                elevation: 0
                md_bg_color: app.c_surface
                line_color: app.c_border
                line_width: dp(0.8)

                MDLabel:
                    text: "On-device storage"
                    font_name: "RobotoLight"
                    font_size: "20sp"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_text

                MDLabel:
                    id: storage_path
                    text: ""
                    font_style: "Caption"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_muted

                MDLabel:
                    id: file_status
                    text: ""
                    markup: True
                    font_style: "Body2"
                    adaptive_height: True
                    theme_text_color: "Custom"
                    text_color: app.c_text

                MDBoxLayout:
                    size_hint_y: None
                    height: dp(52)
                    spacing: dp(12)

                    PillButton:
                        icon: "shield-check-outline"
                        text: "Self-test"
                        filled: True
                        on_release: root.run_self_test()

                    PillButton:
                        icon: "refresh"
                        text: "Refresh"
                        filled: False
                        on_release: root.refresh_storage_status()

    Toast:
        id: toast


<WelcomeScreen>:
    name: "welcome"

    MDBoxLayout:
        orientation: "vertical"
        padding: dp(28)
        spacing: dp(18)
        md_bg_color: app.c_bg

        Widget:

        MDCard:
            size_hint: None, None
            size: dp(120), dp(120)
            radius: [dp(60)]
            elevation: 0
            pos_hint: {"center_x": 0.5}
            md_bg_color: app.c_accent_soft
            line_color: app.c_border
            line_width: dp(0.8)

            MDAnchorLayout:
                MDIcon:
                    size_hint: None, None
                    size: self.font_size + dp(8), self.font_size + dp(8)
                    icon: "hanger"
                    font_size: "60sp"
                    halign: "center"
                    valign: "middle"
                    theme_text_color: "Custom"
                    text_color: app.c_accent

        MDLabel:
            text: "Your private AI stylist"
            font_style: "H4"
            bold: True
            halign: "center"
            adaptive_height: True
            theme_text_color: "Custom"
            text_color: app.c_text

        MDLabel:
            text: "Your photos, measurements and wardrobe stay on this device. No accounts, no servers, no tracking."
            font_style: "Body1"
            halign: "center"
            adaptive_height: True
            theme_text_color: "Custom"
            text_color: app.c_muted

        Widget:
            size_hint_y: None
            height: dp(8)

        PillButton:
            icon: "arrow-right"
            text: "Get started"
            filled: True
            on_release: app.complete_onboarding()

        Widget:


<CameraScreen>:
    name: "camera"

    MDBoxLayout:
        orientation: "vertical"
        md_bg_color: 0, 0, 0, 1

        MDTopAppBar:
            title: "Camera"
            elevation: 0
            md_bg_color: 0, 0, 0, 1
            specific_text_color: 1, 1, 1, 1
            left_action_items: [["arrow-left", lambda x: root.close()]]

        RelativeLayout:
            id: holder

        MDBoxLayout:
            size_hint_y: None
            height: dp(112)
            padding: dp(28), dp(16)
            spacing: dp(16)

            MDIconButton:
                icon: "phone-rotate-portrait"
                theme_icon_color: "Custom"
                icon_color: 1, 1, 1, 1
                pos_hint: {"center_y": 0.5}
                on_release: root.rotate_preview()

            Widget:

            MDIconButton:
                icon: "camera-iris"
                icon_size: "40sp"
                theme_icon_color: "Custom"
                icon_color: app.c_on_accent
                md_bg_color: app.c_accent
                pos_hint: {"center_y": 0.5}
                on_release: root.capture()

            Widget:

            Widget:
                size_hint_x: None
                width: dp(48)


<DockButton>:
    size_hint: None, None
    size: dp(58), dp(50)
    radius: [dp(25)]
    elevation: 0
    md_bg_color: app.c_accent_soft if root.active else (0, 0, 0, 0)
    opacity: 0.7 if self.state == "down" else 1

    canvas.after:
        Color:
            rgba: (app.c_accent[0], app.c_accent[1], app.c_accent[2], 1 if root.active else 0)
        Ellipse:
            pos: self.center_x - dp(2), self.y + dp(5)
            size: dp(4), dp(4)

    MDAnchorLayout:
        MDIcon:
            icon: root.icon_active if root.active else root.icon
            font_size: "23sp"
            size_hint: None, None
            size: dp(28), dp(28)
            halign: "center"
            valign: "middle"
            theme_text_color: "Custom"
            text_color: app.c_text if root.active else app.c_muted


<TopBar>:
    size_hint_y: None
    height: dp(56)
    padding: dp(22), 0, dp(10), 0

    Label:
        text: root.title
        font_name: "RobotoMedium"
        font_size: "15sp"
        color: app.c_text
        halign: "left"
        valign: "middle"
        text_size: self.size

    IconTap:
        icon: app.theme_icon
        font_size: "22sp"
        size_hint: None, None
        size: dp(44), dp(44)
        pos_hint: {"center_y": 0.5}
        color: app.c_text
        on_release: app.toggle_theme()


<FloatingDock@MDCard>:
    size_hint: None, None
    size: dock_row.minimum_width + dp(16), dp(62)
    radius: [dp(31)]
    elevation: 0
    padding: dp(8), dp(6)
    md_bg_color: app.c_surface
    line_color: app.c_border
    line_width: dp(0.8)

    MDBoxLayout:
        id: dock_row
        adaptive_width: True
        spacing: dp(6)

        DockButton:
            zone: "scanner"
            icon: "camera-outline"
            icon_active: "camera"
            active: app.current_zone == "scanner"
            on_release: app.switch_zone("scanner")

        DockButton:
            zone: "closet"
            icon: "hanger"
            icon_active: "hanger"
            active: app.current_zone == "closet"
            on_release: app.switch_zone("closet")

        DockButton:
            zone: "stylist"
            icon: "star-four-points-outline"
            icon_active: "star-four-points"
            active: app.current_zone == "stylist"
            on_release: app.switch_zone("stylist")

        DockButton:
            zone: "profile"
            icon: "account-outline"
            icon_active: "account"
            active: app.current_zone == "profile"
            on_release: app.switch_zone("profile")


<ShellScreen>:
    name: "shell"

    MDFloatLayout:
        md_bg_color: app.c_bg

        MDBoxLayout:
            orientation: "vertical"
            pos_hint: {"x": 0, "y": 0}

            TopBar:
                id: top_bar
                title: "Scanner"

            MDScreenManager:
                id: zones
                transition: FadeTransition(duration=0.18)

                MDScreen:
                    name: "scanner"
                    ScannerZone:
                        id: scanner_zone

                MDScreen:
                    name: "closet"
                    ZonePage:
                        icon: "hanger"
                        headline: "Your closet"
                        body: "A responsive grid of every item you own, filterable by category, colour, occasion and status."
                        module_tag: "COMING NEXT: CLOSET GRID"

                MDScreen:
                    name: "stylist"
                    StylistZone:
                        id: stylist_zone

                MDScreen:
                    name: "profile"
                    ProfileZone:
                        id: profile_zone

        FloatingDock:
            pos_hint: {"center_x": 0.5, "y": 0.018}
"""


# ---------------------------------------------------------------- primitives

# MDCard already inherits ButtonBehavior (via its ripple behaviour), so it
# dispatches on_press / on_release and exposes ``state`` natively.

class Surface(ButtonBehavior, BoxLayout):
    """Lightweight rounded surface with a hairline border (plain Kivy).

    Used for everything that is created in bulk (outfit cards, trend cards,
    rows, chips): KivyMD 1.2 widgets register theme observers whose cost grows
    with every widget ever created, which made rebuilding a deck take seconds.
    """

    bg = ColorProperty([0, 0, 0, 0])
    line = ColorProperty([0, 0, 0, 0])
    line_w = NumericProperty(dp(0.8))
    radius_v = NumericProperty(dp(16))


class IconTap(ButtonBehavior, Label):
    """Tappable Material Design icon (plain Kivy label in the Icons font)."""

    icon = StringProperty("")


class PillButton(MDCard):
    """Full-width pill button styled from the theme tokens."""

    icon = StringProperty("")
    text = StringProperty("")
    filled = BooleanProperty(True)


class ChipButton(Surface):
    """Selectable pill chip - used instead of dropdown menus."""

    icon = StringProperty("")
    text = StringProperty("")
    selected = BooleanProperty(False)
    value = StringProperty("")
    pulse = NumericProperty(0.0)

    def on_release(self) -> None:
        # Micro-interaction: a quick neon ring flash confirms every tap.
        Animation.cancel_all(self, "pulse")
        self.pulse = 1.0
        Animation(pulse=0.0, d=0.45, t="out_quad").start(self)


class Collapsible(MDBoxLayout):
    """Vertical box that collapses to zero height when ``shown`` is False.

    Hidden content ignores touches. It deliberately does NOT use ``disabled``:
    a disabled MDCard restores the colour it had when it was disabled, which
    would leave light-theme colours behind after a dark-mode switch.
    """

    shown = BooleanProperty(True)

    def on_touch_down(self, touch: Any) -> bool:
        return super().on_touch_down(touch) if self.shown else False

    def on_touch_move(self, touch: Any) -> bool:
        return super().on_touch_move(touch) if self.shown else False

    def on_touch_up(self, touch: Any) -> bool:
        return super().on_touch_up(touch) if self.shown else False


class SwatchPill(Surface):
    """Colour dot + consumer colour name + share."""

    hex_color = StringProperty("#888888")
    label = StringProperty("")
    detail = StringProperty("")


class Toast(MDCard):
    """Non-interactive confirmation pill that fades in and out."""

    text = StringProperty("")
    icon = StringProperty("check-circle")

    def on_touch_down(self, touch: Any) -> bool:
        return False

    def on_touch_move(self, touch: Any) -> bool:
        return False

    def on_touch_up(self, touch: Any) -> bool:
        return False

    def flash(self, text: str, icon: str = "check-circle", hold: float = 2.0) -> None:
        Animation.cancel_all(self, "opacity")
        self.text = text
        self.icon = icon
        anim = (Animation(opacity=1, d=0.18, t="out_quad")
                + Animation(opacity=1, d=hold)
                + Animation(opacity=0, d=0.3, t="in_quad"))
        anim.start(self)


# ---------------------------------------------------------------- scanner

class CaptureFrame(MDCard):
    """Rounded capture container with corner guides; shows the cut-out."""

    image_source = StringProperty("")
    has_image = BooleanProperty(False)
    pill_text = StringProperty("")


class ConfirmBadge(ButtonBehavior, MDRelativeLayout):
    """The one-tap 'Confirm Navy Blue Hoodie' badge with a pulsing glow."""

    text = StringProperty("")
    subtext = StringProperty("")
    glow = NumericProperty(0.0)
    _pulse: Optional[Animation] = None

    def start_pulse(self) -> None:
        self.stop_pulse()
        self._pulse = (Animation(glow=1.0, d=1.1, t="in_out_sine")
                       + Animation(glow=0.15, d=1.1, t="in_out_sine"))
        self._pulse.repeat = True
        self._pulse.start(self)

    def stop_pulse(self) -> None:
        if self._pulse is not None:
            self._pulse.cancel(self)
            self._pulse = None
        self.glow = 0.0


class ScanOverlay(ModalView):
    """Full-screen 'AI Scanning Wardrobe...' backdrop with a sweeping scan
    beam over the user's photo, a spinner and live stage text."""

    title = StringProperty("AI Scanning Wardrobe")
    source_image = StringProperty("")
    stage_text = StringProperty("")
    privacy_text = StringProperty("")
    scan_pos = NumericProperty(0.06)
    glow = NumericProperty(0.0)
    active = BooleanProperty(False)
    progress = NumericProperty(0.0)
    percent_text = StringProperty("0%")
    elapsed_text = StringProperty("0.00 S")

    #: Pipeline steps reported by the vision / profile workers.
    TOTAL_STEPS = 5

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._timer_event = None
        self._anims: List[Animation] = []
        self.is_showing = False
        self.opened_at = 0.0
        self._step = 0
        self._done = False

    def on_progress(self, _inst: Any, value: float) -> None:
        self.percent_text = f"{int(round(max(0.0, min(1.0, value)) * 100))}%"

    def on_open(self) -> None:
        self.active = True
        self.scan_pos, self.glow = 0.06, 0.0
        beam = (Animation(scan_pos=0.94, d=1.1, t="in_out_sine")
                + Animation(scan_pos=0.06, d=1.1, t="in_out_sine"))
        beam.repeat = True
        glow = Animation(glow=1.0, d=0.8, t="in_out_sine") + Animation(glow=0.0, d=0.8, t="in_out_sine")
        glow.repeat = True
        self._anims = [beam, glow]
        for anim in self._anims:
            anim.start(self)
        self._timer_event = Clock.schedule_interval(self._tick_timer, 1 / 30.0)

    def on_dismiss(self) -> None:
        self.active = False
        self.is_showing = False
        for anim in self._anims:
            anim.cancel(self)
        self._anims = []
        Animation.cancel_all(self, "progress")
        if self._timer_event is not None:
            self._timer_event.cancel()
            self._timer_event = None

    def _tick_timer(self, _dt: float) -> None:
        if not self._done:
            self.elapsed_text = f"{time.perf_counter() - self.opened_at:.2f} S"

    def set_stage(self, text: str) -> None:
        """Advance one pipeline step: new caption, smoothly filled progress."""
        self.stage_text = text
        self._step += 1
        target = min(0.92, self._step / float(self.TOTAL_STEPS))
        Animation.cancel_all(self, "progress")
        Animation(progress=target, d=0.35, t="out_cubic").start(self)

    def complete(self, processing_ms: Optional[int] = None) -> None:
        """Fill the bar and freeze the timer on the real processing time."""
        self._done = True
        seconds = (processing_ms / 1000.0) if processing_ms is not None else time.perf_counter() - self.opened_at
        self.stage_text = f"Done in {seconds:.2f} s"
        self.elapsed_text = f"{seconds:.2f} S"
        Animation.cancel_all(self, "progress")
        Animation(progress=1.0, d=0.25, t="out_cubic").start(self)

    def show(self, source_image: str, privacy_text: str,
             title: str = "AI Scanning Wardrobe") -> None:
        ext = os.path.splitext(source_image)[1].lower()
        self.source_image = source_image if ext in (".jpg", ".jpeg", ".png", ".bmp") else ""
        self.stage_text = "Preparing"
        self.privacy_text = privacy_text
        self.title = title
        self._step = 0
        self._done = False
        self.progress = 0.0
        self.elapsed_text = "0.00 S"
        if not self.is_showing:
            self.is_showing = True
            self.opened_at = time.perf_counter()
            self.open()


class ScannerZone(MDFloatLayout):
    """Scanner tab: capture -> full-screen AI scan -> cut-out + one-tap badge."""

    has_result = BooleanProperty(False)
    image_source = StringProperty("")
    frame_badge = StringProperty("")
    confirm_text = StringProperty("")
    confidence_text = StringProperty("")
    engine_title = StringProperty("Private on-device AI")
    engine_status = StringProperty("")
    engine_icon = StringProperty("shield-check-outline")

    #: Keep the scan overlay up at least this long so it never just flickers.
    MIN_OVERLAY_SECONDS = 0.7

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.result: Optional[ScanResult] = None
        self.selected_category: Optional[str] = None
        self.show_all_types = False
        self._busy = False
        self._overlay: Optional[ScanOverlay] = None
        self._last_source = "camera"

    def on_kv_post(self, base_widget: Any) -> None:
        Clock.schedule_once(lambda _dt: self.refresh_engine_status(), 0)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _app() -> "WardrobeApp":
        return MDApp.get_running_app()  # type: ignore[return-value]

    def _cutout_cfg(self) -> Dict[str, Any]:
        cfg = self._app().storage.get_setting("cutout", {})
        return dict(cfg) if isinstance(cfg, dict) else {}

    def refresh_engine_status(self) -> None:
        app = self._app()
        if app is None or app.storage is None:
            return
        if not CV_AVAILABLE:
            self.engine_title = "Vision engine unavailable"
            self.engine_status = f"Install OpenCV + NumPy ({CV_IMPORT_ERROR})"
            self.engine_icon = "alert-circle-outline"
            return
        cfg = self._cutout_cfg()
        provider = cfg.get("provider", "local")
        local = app.vision.status_text() if app.vision is not None else ""
        if provider != "local" and cloud_provider_ready(cfg):
            self.engine_title = f"Cloud cut-out · {CLOUD_PROVIDERS.get(provider, provider)}"
            self.engine_status = f"Photos are uploaded for background removal. Fallback: {local}"
            self.engine_icon = "cloud-upload-outline"
        else:
            self.engine_title = "Private on-device AI"
            self.engine_status = local
            self.engine_icon = "shield-check-outline"

    def _vision_ready(self) -> bool:
        if CV_AVAILABLE and self._app().vision is not None:
            return True
        self._app().show_dialog(
            "Vision engine missing",
            "OpenCV and NumPy are required for scanning.\n\n"
            "Desktop: pip install -r requirements.txt\n"
            f"Details: {CV_IMPORT_ERROR or 'not initialised'}",
        )
        return False

    # -- inputs -----------------------------------------------------------
    def open_camera(self) -> None:
        if self._busy or not self._vision_ready():
            return
        self._last_source = "camera"
        self._app().open_camera_for("scanner")

    def open_gallery(self) -> None:
        if self._busy or not self._vision_ready():
            return
        self._last_source = "gallery"
        self._app().pick_image(
            lambda path, cleanup, orient: self.start_processing(path, cleanup, orient)
        )

    # -- processing -------------------------------------------------------
    def start_processing(self, path: str, cleanup: bool = False, exif_orientation: int = 1) -> None:
        if self._busy:
            return
        if self.result is not None:
            self._delete_pending_files()
            self._reset_result()
        self._busy = True

        cfg = self._cutout_cfg()
        provider = cfg.get("provider", "local")
        if provider != "local" and cloud_provider_ready(cfg):
            privacy = f"Uploading to {CLOUD_PROVIDERS.get(provider, provider)} for the cut-out"
        else:
            privacy = "Processed privately on this device"

        if self._overlay is None:
            self._overlay = ScanOverlay()
        self._overlay.show(path, privacy)

        threading.Thread(
            target=self._worker, args=(path, cleanup, exif_orientation, cfg), daemon=True
        ).start()

    def _post_progress(self, text: str) -> None:
        def _apply(_dt: float) -> None:
            if self._overlay is not None:
                self._overlay.set_stage(text)

        Clock.schedule_once(_apply, 0)

    def _worker(self, path: str, cleanup: bool, exif_orientation: int, cfg: Dict[str, Any]) -> None:
        app = self._app()
        try:
            result = app.vision.process_file(path, app.storage, self._post_progress,
                                             exif_orientation, cfg)
            Clock.schedule_once(lambda _dt: self._finish(lambda: self._on_result(result), result.processing_ms), 0)
        except VisionError as exc:
            message = str(exc)
            Clock.schedule_once(lambda _dt: self._finish(lambda: self._on_error(message)), 0)
        except Exception as exc:  # noqa: BLE001 - never let the worker die silently
            Logger.exception(f"{LOG_TAG}: scan failed")
            message = f"Unexpected error while scanning: {exc}"
            Clock.schedule_once(lambda _dt: self._finish(lambda: self._on_error(message)), 0)
        finally:
            if cleanup:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _finish(self, then: Callable[[], None], processing_ms: Optional[int] = None) -> None:
        """Close the overlay (after its minimum display time), then continue."""
        overlay = self._overlay
        elapsed = time.perf_counter() - overlay.opened_at if overlay is not None else 1e9
        delay = max(0.0, self.MIN_OVERLAY_SECONDS - elapsed)
        if processing_ms is not None and overlay is not None:
            overlay.complete(processing_ms)
            delay = max(delay, 0.45)  # let "Done in 0.62 s" register

        def _close(_dt: float) -> None:
            if self._overlay is not None and self._overlay.is_showing:
                self._overlay.dismiss()
            self._busy = False
            then()

        Clock.schedule_once(_close, delay)

    def _on_error(self, message: str) -> None:
        self.refresh_engine_status()
        self._app().show_dialog("Scan failed", message)

    def _on_result(self, result: ScanResult) -> None:
        self.result = result
        self.selected_category = result.category
        self.show_all_types = False
        self.image_source = self._app().storage.path(result.cutout_rel)

        seconds = result.processing_ms / 1000.0
        self.frame_badge = f"{segmenter_label(result.segmenter)} · {seconds:.1f} s"
        threshold = float(self._app().storage.get_setting("scanner.confirm_threshold", 0.75))
        pct = f"{result.confidence:.0%}"
        if result.confidence >= threshold:
            self.confidence_text = f"AI match {pct}. Tap the badge to save, or pick another type."
        else:
            self.confidence_text = f"AI match {pct} - worth a glance. Pick the right type if needed."

        self._render_choices()
        self._render_swatches()
        self._update_confirm_text()
        self.has_result = True
        self.ids.confirm_badge.start_pulse()
        self.refresh_engine_status()
        Logger.info(
            f"{LOG_TAG}: scan {result.item_id} -> {self.confirm_text} "
            f"({result.confidence:.0%}) seg={result.segmenter} in {result.processing_ms} ms"
        )

    # -- result rendering ---------------------------------------------------
    def _update_confirm_text(self) -> None:
        result = self.result
        if result is None or self.selected_category is None:
            self.confirm_text = ""
            return
        color = result.primary_color_name
        name = f"{color} {self.selected_category}" if color else self.selected_category
        self.confirm_text = f"Confirm {name}"

    def _render_choices(self) -> None:
        grid = self.ids.choice_grid
        grid.clear_widgets()
        result = self.result
        if result is None:
            return
        if self.show_all_types:
            options = list(CATEGORIES)
        else:
            ranked = [c for c, _s in result.ranking]
            options = ranked[:4]
            if self.selected_category and self.selected_category not in options:
                options = [self.selected_category] + options[:3]
        for name in options:
            score = result.score_for(name)
            chip = ChipButton(
                text=f"{name}  {score:.0%}" if not self.show_all_types else name,
                value=name,
                icon="check" if name == self.selected_category else "",
                selected=name == self.selected_category,
            )
            chip.bind(on_release=lambda btn: self._select_category(btn.value))
            grid.add_widget(chip)
        toggle = ChipButton(
            text="Fewer" if self.show_all_types else "More types",
            icon="chevron-up" if self.show_all_types else "dots-horizontal",
        )
        toggle.bind(on_release=lambda _btn: self._toggle_all_types())
        grid.add_widget(toggle)

    def _render_swatches(self) -> None:
        row = self.ids.swatch_row
        row.clear_widgets()
        result = self.result
        if result is None:
            return
        for color in result.colors:
            row.add_widget(SwatchPill(
                hex_color=color["hex"],
                label=color["name"],
                detail=f"{color['hex']} · {color['share']:.0%}",
            ))
        meta = [f"{result.pattern.capitalize()} pattern", f"{result.color_family} tones"]
        if result.shape:
            meta.append(f"shape ratio {result.shape.get('aspect', 0):.2f}")
        text = "  ·  ".join(meta)
        if result.notes:
            text += "\n" + " ".join(result.notes)
        self.ids.meta_label.text = text

    def _select_category(self, category: str) -> None:
        if category not in CATEGORY_META:
            return
        self.selected_category = category
        self._update_confirm_text()
        self._render_choices()

    def _toggle_all_types(self) -> None:
        self.show_all_types = not self.show_all_types
        self._render_choices()

    # -- confirm / discard ------------------------------------------------
    def confirm(self) -> None:
        """One tap: build the record and atomically cache it in items.json."""
        result = self.result
        if result is None or self.selected_category is None:
            return
        app = self._app()
        record = build_item_record(
            result,
            self.selected_category,
            True,  # the user confirmed this exact label by tapping the badge
            None,  # price can be added later from the Closet - no forms here
            str(app.storage.get_setting("currency", "USD")),
        )

        def _add(doc: Dict[str, Any]) -> None:
            items = doc.get("items")
            if not isinstance(items, dict):
                items = {}
                doc["items"] = items
            items[record["item_id"]] = record

        try:
            doc = app.storage.update("items", _add)
        except StorageError as exc:
            app.show_dialog("Save failed", str(exc))
            return

        total = len(doc.get("items", {}))
        self.result = None  # files now belong to a saved item
        self._reset_result()
        self.ids.toast.flash(f"Saved {record['display_name']} · {total} in wardrobe")

    def discard(self) -> None:
        self._delete_pending_files()
        self._reset_result()

    def retake(self) -> None:
        source = self._last_source
        self.discard()
        if source == "gallery":
            self.open_gallery()
        else:
            self.open_camera()

    def _delete_pending_files(self) -> None:
        result = self.result
        if result is None:
            return
        storage = self._app().storage
        for rel in (result.original_rel, result.cutout_rel, result.thumb_rel):
            try:
                os.remove(storage.path(rel))
            except OSError:
                pass
        self.result = None

    def _reset_result(self) -> None:
        self.ids.confirm_badge.stop_pulse()
        self.result = None
        self.selected_category = None
        self.show_all_types = False
        self.has_result = False
        self.image_source = ""
        self.frame_badge = ""
        self.confirm_text = ""
        self.confidence_text = ""
        self.ids.choice_grid.clear_widgets()
        self.ids.swatch_row.clear_widgets()
        self.ids.meta_label.text = ""


# ---------------------------------------------------------------- camera

class CameraScreen(MDScreen):
    """Full-screen in-app camera. Captures exactly what the preview shows."""

    angle = NumericProperty(0)  # clockwise degrees: 0 / 90 / 180 / 270

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._camera = None
        self._rotate: Optional[Rotate] = None
        self._watchdog = None
        self._capturing = False

    @staticmethod
    def _app() -> "WardrobeApp":
        return MDApp.get_running_app()  # type: ignore[return-value]

    def on_enter(self, *args: Any) -> None:
        self.angle = int(self._app().storage.get_setting("camera.rotation", 0)) % 360
        if IS_ANDROID:
            from android.permissions import (  # type: ignore[import-not-found]
                Permission,
                check_permission,
                request_permissions,
            )
            if not check_permission(Permission.CAMERA):
                request_permissions([Permission.CAMERA], self._on_permission_result)
                return
        self._start_camera()

    def on_leave(self, *args: Any) -> None:
        self._stop_camera()

    def _on_permission_result(self, _permissions: List[str], grants: List[bool]) -> None:
        granted = bool(grants) and all(grants)

        def _apply(_dt: float) -> None:
            if granted:
                self._start_camera()
            else:
                self._fail("Camera permission was denied. You can still upload a photo, "
                           "or allow the camera in Android Settings > Apps.")

        Clock.schedule_once(_apply, 0)

    def _start_camera(self) -> None:
        if self.manager is None or self.manager.current != self.name:
            return
        self._stop_camera()
        try:
            from kivy.uix.camera import Camera
            cam = Camera(index=0, resolution=(1280, 720), play=True,
                         fit_mode="contain", size_hint=(None, None))
        except Exception as exc:  # noqa: BLE001 - provider-specific failures
            Logger.exception(f"{LOG_TAG}: camera init failed")
            self._fail(f"Could not open the camera ({exc}). Try Upload photo instead.")
            return

        with cam.canvas.before:
            PushMatrix()
            self._rotate = Rotate(angle=-self.angle, origin=(0, 0))
        with cam.canvas.after:
            PopMatrix()

        self._camera = cam
        holder = self.ids.holder
        holder.add_widget(cam)
        holder.bind(size=self._layout_camera)
        self._layout_camera()
        self._watchdog = Clock.schedule_once(self._check_frames, 5.0)

    def _layout_camera(self, *_args: Any) -> None:
        cam = self._camera
        if cam is None:
            return
        hw, hh = self.ids.holder.size
        # Rotated 90/270: give the camera swapped dimensions so it fills the
        # holder once rotated.
        cam.size = (hh, hw) if self.angle in (90, 270) else (hw, hh)
        cam.center = (hw / 2.0, hh / 2.0)  # RelativeLayout -> local coordinates
        if self._rotate is not None:
            self._rotate.origin = cam.center
            self._rotate.angle = -self.angle  # Kivy rotates counter-clockwise

    def _check_frames(self, _dt: float) -> None:
        self._watchdog = None
        if self._camera is not None and self._camera.texture is None:
            self._fail("The camera is not delivering any frames. Check that no other app "
                       "is using it, or upload a photo instead.")

    def rotate_preview(self) -> None:
        self.angle = (int(self.angle) + 90) % 360
        self._app().storage.set_setting("camera.rotation", int(self.angle))
        self._layout_camera()

    def capture(self) -> None:
        cam = self._camera
        if self._capturing or cam is None or cam.texture is None:
            return
        self._capturing = True
        app = self._app()
        path = app.storage.path("cache", f"capture_{uuid.uuid4().hex}.png")
        try:
            # Exporting the holder captures exactly what the user sees,
            # including the rotation, independent of sensor orientation.
            self.ids.holder.export_to_png(path, scale=1 if IS_MOBILE else 2)
        except Exception as exc:  # noqa: BLE001 - GL/FBO failures
            Logger.exception(f"{LOG_TAG}: capture failed")
            self._capturing = False
            app.show_dialog("Camera", f"Could not capture the photo: {exc}")
            return
        self._capturing = False
        self.close()
        app.on_camera_capture(path)

    def _fail(self, message: str) -> None:
        self.close()
        self._app().show_dialog("Camera", message)

    def close(self) -> None:
        self._stop_camera()
        if self.manager is not None and self.manager.current == self.name:
            self.manager.current = "shell"

    def _stop_camera(self) -> None:
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        cam = self._camera
        if cam is None:
            return
        self._camera = None
        self._rotate = None
        try:
            cam.play = False
            core = getattr(cam, "_camera", None)
            # Release the device so the webcam light / Android camera turns off.
            if core is not None and hasattr(core, "_release_camera"):
                core._release_camera()
            device = getattr(core, "_device", None)
            if device is not None and hasattr(device, "release"):
                device.release()
        except Exception:  # noqa: BLE001 - best effort cleanup
            Logger.exception(f"{LOG_TAG}: camera release failed")
        holder = self.ids.holder
        holder.unbind(size=self._layout_camera)
        holder.remove_widget(cam)


# ---------------------------------------------------------------- other zones

class TopBar(BoxLayout):
    """Minimal header: left-aligned title and a theme toggle (plain Kivy)."""

    title = StringProperty("")


class DockButton(MDCard):
    """Icon button in the floating dock (outline icon -> filled when active)."""

    zone = StringProperty("")
    icon = StringProperty("")
    icon_active = StringProperty("")
    active = BooleanProperty(False)


class ZoneCard(MDCard):
    """Placeholder card describing a feature zone."""

    icon = StringProperty("help-circle")
    headline = StringProperty("")
    body = StringProperty("")
    module_tag = StringProperty("")


class ZonePage(MDBoxLayout):
    """Scrollable page holding a ZoneCard; later modules replace these."""

    icon = StringProperty("help-circle")
    headline = StringProperty("")
    body = StringProperty("")
    module_tag = StringProperty("")


class CutoutSettingsCard(MDCard):
    """Background-removal engine picker (chips, no dropdowns)."""

    field_hint = StringProperty("")
    field_secret = BooleanProperty(True)
    privacy_note = StringProperty("")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.provider = "local"
        self._cfg: Dict[str, Any] = {}

    def on_kv_post(self, base_widget: Any) -> None:
        Clock.schedule_once(lambda _dt: self.load(), 0)

    @staticmethod
    def _app() -> "WardrobeApp":
        return MDApp.get_running_app()  # type: ignore[return-value]

    def load(self) -> None:
        app = self._app()
        if app is None or app.storage is None:
            return
        cfg = app.storage.get_setting("cutout", {})
        self._cfg = dict(cfg) if isinstance(cfg, dict) else {}
        provider = self._cfg.get("provider", "local")
        self.provider = provider if provider in CLOUD_PROVIDERS else "local"
        self._render()

    def _field_key(self) -> Optional[str]:
        return {
            "removebg": "removebg_api_key",
            "photoroom": "photoroom_api_key",
            "rembg_server": "server_url",
        }.get(self.provider)

    def _render(self) -> None:
        chips = self.ids.provider_chips
        chips.clear_widgets()
        icons = {"local": "shield-check-outline", "removebg": "cloud-outline",
                 "photoroom": "cloud-outline", "rembg_server": "server-network"}
        for key, label in CLOUD_PROVIDERS.items():
            chip = ChipButton(text=label, value=key, icon=icons[key], selected=key == self.provider)
            chip.bind(on_release=lambda btn: self._choose(btn.value))
            chips.add_widget(chip)

        field_key = self._field_key()
        if self.provider == "rembg_server":
            self.field_hint = "Server URL, e.g. http://192.168.1.20:7000"
            self.field_secret = False
        elif field_key:
            self.field_hint = f"{CLOUD_PROVIDERS[self.provider]} API key"
            self.field_secret = True
        else:
            self.field_hint = ""
        self.ids.cred_field.text = str(self._cfg.get(field_key, "")) if field_key else ""

        notes = {
            "local": "Private: photos never leave this phone.",
            "removebg": "Uploads each photo to remove.bg. Free accounts get low-resolution "
                        "previews; the key is stored in this app's private storage.",
            "photoroom": "Uploads each photo to Photoroom. Paid API; the key is stored in "
                         "this app's private storage.",
            "rembg_server": "Uploads each photo to your own rembg server (run: rembg s). "
                            "Private if it stays on your network.",
        }
        self.privacy_note = notes[self.provider]

    def _choose(self, provider: str) -> None:
        field_key = self._field_key()
        if field_key:
            self._cfg[field_key] = self.ids.cred_field.text.strip()
        self.provider = provider
        self._render()

    def save(self) -> None:
        app = self._app()
        field_key = self._field_key()
        if field_key:
            self._cfg[field_key] = self.ids.cred_field.text.strip()
        self._cfg["provider"] = self.provider
        if self.provider != "local" and not cloud_provider_ready(self._cfg):
            app.show_dialog(
                "Background removal",
                "Add your API key (or a server URL starting with http://) first. "
                "Until then the scanner keeps using on-device processing.",
            )
        app.storage.set_setting("cutout", self._cfg)
        app.scanner_zone().refresh_engine_status()
        app.show_dialog("Background removal",
                        f"Engine set to {CLOUD_PROVIDERS[self.provider]}.")


# ---------------------------------------------------------------- stylist UI

class ColorDot(Widget):
    rgba = ColorProperty([0.5, 0.5, 0.5, 1.0])


class ScoreRing(MDRelativeLayout):
    value = NumericProperty(0.0)
    label = StringProperty("")


class ItemThumb(MDCard):
    source = StringProperty("")
    caption = StringProperty("")
    layer_text = StringProperty("")
    swatch = StringProperty("")


class ReasonPill(Surface):
    text = StringProperty("")


RULE_SHORT: Dict[str, str] = {
    "monochromatic": "MONO", "analogous": "ANALOGOUS", "complementary": "COMPLEMENT",
    "split_complementary": "SPLIT", "triadic": "TRIADIC", "neutral_anchor": "ANCHOR",
    "neutral": "NEUTRAL", "clash": "BOLD", "shuffle": "SHUFFLE",
}


def _garment_source(storage: StorageManager, piece: Dict[str, Any]) -> str:
    """Best image for a garment: full cut-out, then thumbnail, else ''."""
    images = piece.get("images") or {}
    for key in ("cutout", "thumb"):
        rel = images.get(key)
        if rel:
            path = storage.path(rel)
            if os.path.isfile(path):
                return path
    return ""


def _ink_for(hex_code: str) -> List[float]:
    """Readable label colour on top of a garment colour block."""
    r, g, b = hex_to_rgb01(hex_code or "#808080")
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return [0, 0, 0, 0.7] if luminance > 0.55 else [1, 1, 1, 0.85]


class GarmentTile(Surface):
    """Colour-block stand-in for a garment without a photo."""

    hex_color = StringProperty("#808080")
    caption = StringProperty("")
    ink = ColorProperty([1, 1, 1, 1])


class MissingFrame(RelativeLayout):
    """Dashed placeholder for a piece you don't own yet."""

    hex_color = StringProperty("#808080")
    caption = StringProperty("")


class FlatLay(RelativeLayout):
    """High-fashion flat lay: garment cut-outs floating in overlapping layer
    frames over a soft spotlight (outer layer behind, inner layers stacked
    forward, bottoms to the side, shoes in front)."""

    spot_alpha = NumericProperty(0.05)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frames: List[Any] = []

    @staticmethod
    def _layout(pieces: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Tuple[float, float, float, float]]]:
        tops = [p for p in pieces if _slot(p) == "top"]
        outer = [t for t in tops if _level(t) == 3]
        inner = sorted([t for t in tops if _level(t) != 3], key=lambda t: -(_level(t) or 0))
        bottoms = [p for p in pieces if _slot(p) == "bottom"]
        dresses = [p for p in pieces if _slot(p) == "full_body"]
        shoes = [p for p in pieces if _slot(p) == "footwear"]
        placed: List[Tuple[Dict[str, Any], Tuple[float, float, float, float]]] = []
        if dresses:
            for o in outer[:1]:
                placed.append((o, (0.02, 0.30, 0.46, 0.62)))
            placed.append((dresses[0], (0.24, 0.06, 0.52, 0.84)))
        else:
            for o in outer[:1]:
                placed.append((o, (0.02, 0.34, 0.50, 0.60)))
            start_x = 0.12 if outer else 0.04
            for i, t in enumerate(inner[:3]):
                placed.append((t, (start_x + 0.09 * i, 0.40 - 0.045 * i, 0.46, 0.52)))
            for b in bottoms[:1]:
                placed.append((b, (0.52, 0.03, 0.44, 0.78)))
        for s in shoes[:1]:
            placed.append((s, (0.05, 0.02, 0.30, 0.22)))
        return placed

    def compose(self, pieces: List[Dict[str, Any]], storage: StorageManager) -> None:
        from kivy.uix.image import Image
        for frame in self._frames:
            self.remove_widget(frame)
        self._frames = []
        for piece, (x, y, w, h) in self._layout(pieces):
            source = _garment_source(storage, piece)
            if source:
                frame = Image(source=source, fit_mode="contain", nocache=True,
                              size_hint=(w, h), pos_hint={"x": x, "y": y})
            else:
                hex_code = str(piece.get("primary_hex") or "#808080")
                frame = GarmentTile(hex_color=hex_code, caption=str(piece.get("category", "")).upper(),
                                    ink=_ink_for(hex_code), size_hint=(w * 0.74, h * 0.74),
                                    pos_hint={"x": x + w * 0.13, "y": y + h * 0.13})
            self.add_widget(frame)
            self._frames.append(frame)

    def play_intro(self) -> None:
        """Staggered fade-in of the layers - plays when a card comes into view."""
        for i, frame in enumerate(self._frames):
            Animation.cancel_all(frame, "opacity")
            frame.opacity = 0.0
            (Animation(opacity=0.0, d=0.06 * i) + Animation(opacity=1.0, d=0.32, t="out_quad")).start(frame)


class LookCollage(RelativeLayout):
    """Mini lookbook collage: owned pieces as cut-outs, missing ones dashed."""

    spot_alpha = NumericProperty(0.05)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._frames: List[Any] = []

    def compose(self, parts: List[Dict[str, Any]], storage: StorageManager) -> None:
        from kivy.uix.image import Image
        for frame in self._frames:
            self.remove_widget(frame)
        self._frames = []
        n = max(1, len(parts))
        w, h = {1: (0.6, 0.9), 2: (0.5, 0.88), 3: (0.38, 0.84)}.get(n, (0.3, 0.8))
        # Owned pieces first, missing (dashed) frames last so they sit on top.
        order = sorted(range(len(parts)), key=lambda k: parts[k].get("piece") is None)
        for slot_index, i in enumerate(order):
            part = parts[i]
            x = 0.02 + slot_index * ((0.96 - w) / (n - 1)) if n > 1 else 0.2
            y = 0.04 + (0.07 if slot_index % 2 else 0.0)
            piece = part.get("piece")
            hex_code = part.get("hex", "#808080")
            if piece is not None:
                source = _garment_source(storage, piece)
                if source:
                    frame = Image(source=source, fit_mode="contain", nocache=True,
                                  size_hint=(w, h), pos_hint={"x": x, "y": y})
                else:
                    frame = GarmentTile(hex_color=str(piece.get("primary_hex") or hex_code),
                                        caption=str(part.get("category", "")).upper(), ink=_ink_for(hex_code),
                                        size_hint=(w * 0.8, h * 0.8), pos_hint={"x": x + w * 0.1, "y": y + h * 0.1})
            else:
                frame = MissingFrame(hex_color=hex_code, caption=str(part.get("category", "")).upper(),
                                     size_hint=(w * 0.82, h * 0.82), pos_hint={"x": x + w * 0.09, "y": y + h * 0.09})
            self.add_widget(frame)
            self._frames.append(frame)


class ScoreMeter(RelativeLayout):
    value = NumericProperty(0.0)
    label = StringProperty("")
    rule = StringProperty("")


class InteractionChip(Surface):
    """Pill chip with a neon ring flash on every tap."""

    icon = StringProperty("")
    text = StringProperty("")
    active = BooleanProperty(False)
    pulse = NumericProperty(0.0)

    def on_release(self) -> None:
        Animation.cancel_all(self, "pulse")
        self.pulse = 1.0
        Animation(pulse=0.0, d=0.5, t="out_quad").start(self)


class OutfitCard(Surface):
    """Lookbook card: floating flat lay, score meter, reasons, wear + heart."""

    title = StringProperty("")
    subtitle = StringProperty("")
    kicker = StringProperty("")
    score = NumericProperty(0.0)
    score_text = StringProperty("")
    rule_short = StringProperty("")
    saved = BooleanProperty(False)
    worn = BooleanProperty(False)
    shuffle = BooleanProperty(False)
    heart_size = NumericProperty(dp(26))
    outfit = ObjectProperty(None, allownone=True)
    pending_pieces = ObjectProperty(None, allownone=True)

    __events__ = ("on_wear", "on_save")

    def on_wear(self) -> None:
        pass

    def on_save(self) -> None:
        pass

    def pop_heart(self) -> None:
        Animation.cancel_all(self, "heart_size")
        (Animation(heart_size=dp(34), d=0.09, t="out_quad")
         + Animation(heart_size=dp(26), d=0.18, t="out_back")).start(self)


class LookCard(Surface):
    """Trend discovery card: collage, colour-dot chain, one-tap action chip."""

    title = StringProperty("")
    subtitle = StringProperty("")
    cta = StringProperty("")
    cta_active = BooleanProperty(False)
    pulse = NumericProperty(0.0)
    trend = ObjectProperty(None, allownone=True)
    match = ObjectProperty(None, allownone=True)

    __events__ = ("on_chip",)

    def on_chip(self) -> None:
        Animation.cancel_all(self, "pulse")
        self.pulse = 1.0
        Animation(pulse=0.0, d=0.5, t="out_quad").start(self)

    def set_dots(self, hexes: List[str]) -> None:
        box = self.ids.dots
        box.clear_widgets()
        for hx in hexes[:5]:
            dot = ColorDot(rgba=get_color_from_hex(hx))
            dot.size = (dp(14), dp(14))
            box.add_widget(dot)


class LookDetail(ModalView):
    """Popup breakdown of a trend look: each piece, owned or missing."""

    kicker = StringProperty("")
    title = StringProperty("")
    subtitle = StringProperty("")
    primary_text = StringProperty("")
    primary_icon = StringProperty("check")

    __events__ = ("on_primary",)

    def on_primary(self) -> None:
        pass


class ListRow(Surface):
    """Compact row: colour dots, title/subtitle, optional badge and action."""

    title = StringProperty("")
    subtitle = StringProperty("")
    badge = StringProperty("")
    badge_on = BooleanProperty(False)
    action_icon = StringProperty("")
    payload = ObjectProperty(None, allownone=True)

    __events__ = ("on_action",)

    def on_action(self) -> None:
        pass

    def set_swatches(self, hexes: List[str]) -> None:
        box = self.ids.dots
        box.clear_widgets()
        for hx in hexes[:4]:
            box.add_widget(ColorDot(rgba=get_color_from_hex(hx)))


def _short_date(iso: str) -> str:
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%a %d %b")
    except ValueError:
        return iso


class StylistZone(MDFloatLayout):
    """AI Stylist tab: swipeable outfit deck, weather, trends, gaps, packing."""

    headline = StringProperty("Today's looks")
    subline = StringProperty("")
    weather_text = StringProperty("Weather off")
    weather_icon = StringProperty("cloud-off-outline")
    weather_live = BooleanProperty(False)
    weather_on = BooleanProperty(False)
    trends_on = BooleanProperty(False)
    has_outfits = BooleanProperty(False)
    has_trends = BooleanProperty(False)
    empty_title = StringProperty("No outfits yet")
    empty_body = StringProperty("Scan at least one top and one bottom.")
    trend_caption = StringProperty("")
    gap_caption = StringProperty("")
    pack_days = NumericProperty(5)
    pack_days_text = StringProperty("5 days")
    pack_summary = StringProperty("")
    occasion = StringProperty("Casual")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._items: Dict[str, Dict[str, Any]] = {}
        self._outfits: List[Dict[str, Any]] = []
        self._weather: Optional[WeatherContext] = None
        self._pack_occasions = {"Casual"}
        self._busy: set = set()

    def on_kv_post(self, base_widget: Any) -> None:
        Clock.schedule_once(lambda _dt: self.refresh(), 0)

    def on_pack_days(self, _inst: Any, value: float) -> None:
        self.pack_days_text = f"{int(value)} day{'s' if int(value) != 1 else ''}"

    @staticmethod
    def _app() -> "WardrobeApp":
        return MDApp.get_running_app()  # type: ignore[return-value]

    def _run(self, key: str, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        """Run ``work`` on a background thread once at a time per ``key``;
        deliver its result (or exception) to ``done`` on the UI thread."""
        if key in self._busy:
            return
        self._busy.add(key)

        def worker() -> None:
            try:
                result: Any = work()
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                Logger.exception(f"{LOG_TAG}: {key} failed")
                result = exc

            def finish(_dt: float) -> None:
                self._busy.discard(key)
                done(result)

            Clock.schedule_once(finish, 0)

        threading.Thread(target=worker, daemon=True).start()

    # -- refresh ------------------------------------------------------------
    def refresh(self) -> None:
        app = self._app()
        if app is None or app.storage is None:
            return
        st = app.storage
        self.weather_on = bool(st.get_setting("network.weather_enabled", False))
        self.trends_on = bool(st.get_setting("network.trends_enabled", False))
        loc = st.load("profile").get("location") or {}
        if loc.get("city") and not self.ids.city_field.text:
            self.ids.city_field.text = str(loc["city"]).split(",")[0]
        if not self.ids.pack_start.text:
            from datetime import date, timedelta
            self.ids.pack_start.text = (date.today() + timedelta(days=1)).isoformat()
        self._render_occasions()
        self._render_pack_occasions()
        cached, _msg = load_weather(st, allow_network=False)
        # Re-visiting the tab is instant unless the wardrobe, profile, trends,
        # wishlist, weather or toggles actually changed since the last build.
        signature = self._data_signature(cached)
        if signature != getattr(self, "_last_signature", None):
            self._last_signature = signature
            self._load_items()
            self._apply_weather(cached, "")
            self.generate()
            self.render_trends()
            self.render_feeds()
        if self.weather_on:
            self._fetch_weather()
        if self.trends_on:
            self._sync_trends(force=False, quiet=True)

    def _data_signature(self, weather: Optional[WeatherContext]) -> Tuple[Any, ...]:
        st = self._app().storage
        stamps = []
        for name in ("items", "profile", "trends", "wishlist"):
            try:
                stamps.append(os.path.getmtime(st.doc_path(name)))
            except OSError:
                stamps.append(None)
        weather_key = (round(weather.temp), weather.wet) if weather is not None else None
        return tuple(stamps) + (self.weather_on, self.trends_on, weather_key)

    def _load_items(self) -> None:
        items = self._app().storage.load("items").get("items") or {}
        self._items = {k: v for k, v in items.items() if isinstance(v, dict)}

    def _ctx(self) -> StyleContext:
        st = self._app().storage
        profile = st.load("profile")
        skin = profile.get("skin") or {}
        body = profile.get("body") or {}
        return StyleContext(occasion=self.occasion, weather=self._weather,
                            palette=profile.get("palette") or {},
                            undertone_bias=str(skin.get("undertone_bias", "")),
                            body_shape=body.get("shape"), recent_item_ids=recent_item_ids(st))

    # -- occasion chips -------------------------------------------------------
    def _render_occasions(self) -> None:
        box = self.ids.occasion_chips
        if len(box.children) != len(OCCASIONS):
            box.clear_widgets()
            for occ in OCCASIONS:
                chip = ChipButton(text=OCCASION_LABELS.get(occ, occ), value=occ)
                chip.bind(on_release=lambda btn: self.select_occasion(btn.value))
                box.add_widget(chip)
        for chip in box.children:  # update in place - no widget churn
            chip.selected = chip.value == self.occasion
            chip.icon = "check" if chip.selected else ""

    def select_occasion(self, occasion: str) -> None:
        if occasion == self.occasion:
            return
        self.occasion = occasion
        self._render_occasions()
        self.generate()

    # -- deck -----------------------------------------------------------------
    def generate(self, shuffle: bool = False) -> None:
        self._load_items()
        items = list(self._items.values())
        outfits = generate_outfits(items, self._ctx(), limit=10, shuffle=shuffle,
                                   seed=int(time.time() * 1000) if shuffle else None)
        self._render_deck(outfits)
        slots = {_slot(i) for i in items if i.get("status", "active") == "active"}
        if not outfits:
            if "top" not in slots or not ({"bottom", "full_body"} & slots):
                self.empty_title = "Your stylist needs clothes"
                self.empty_body = "Scan at least one top and one bottom in the Scanner tab."
            else:
                w = self._weather
                where = f" at {round(w.temp)}°C" if w else ""
                self.empty_title = "Nothing fits these rules"
                self.empty_body = (f"No {OCCASION_LABELS.get(self.occasion, self.occasion)} outfit works{where}. "
                                   "Try another occasion, or tap Virtual shuffle.")
        label = OCCASION_LABELS.get(self.occasion, self.occasion)
        self.headline = "Shuffled looks" if shuffle else f"{label} looks"
        self.subline = (f"{len(outfits)} combinations · layer physics on · rules off" if shuffle
                        else f"{len(outfits)} ranked combinations from {len(items)} pieces")

    def shuffle(self) -> None:
        self.generate(shuffle=True)
        if self.has_outfits:
            self.ids.toast.flash("Rules off - high-contrast shuffle", icon="shuffle-variant")

    def _render_deck(self, outfits: List[Dict[str, Any]]) -> None:
        """Show ``outfits`` in the deck, re-using existing card widgets.

        Cards are pooled: Shuffle / Refresh only update text, colours and the
        (lazily composed) flat lay of cards that already exist. New cards are
        created only when the deck grows - the first two immediately, the rest
        one per frame so the UI never freezes.
        """
        from datetime import date
        deck = self.ids.deck
        stored = self._app().storage.load("outfits").get("outfits") or {}
        today = date.today().isoformat()
        spare: List[OutfitCard] = getattr(self, "_spare_cards", [])
        self._spare_cards = spare

        def state_of(outfit: Dict[str, Any]) -> Tuple[bool, bool]:
            entry = stored.get(outfit["outfit_id"]) or {}
            return bool(entry.get("saved")), entry.get("last_worn") == today

        self._deck_token = getattr(self, "_deck_token", 0) + 1
        token = self._deck_token
        existing = list(deck.slides)
        for card, outfit in zip(existing, outfits):
            self._fill_card(card, outfit, *state_of(outfit))
        for card in existing[len(outfits):]:
            deck.remove_widget(card)
            spare.append(card)

        missing = list(outfits[len(existing):])
        immediate, later = missing[:2], missing[2:]
        for outfit in immediate:
            deck.add_widget(self._fill_card(spare.pop() if spare else self._new_card(), outfit, *state_of(outfit)))

        def add_next(_dt: float) -> None:
            if token != self._deck_token or not later:
                return
            outfit = later.pop(0)
            card = spare.pop() if spare else self._new_card()
            deck.add_widget(self._fill_card(card, outfit, *state_of(outfit)))
            self.on_deck_index()
            if later:
                Clock.schedule_once(add_next, 0)

        if later:
            Clock.schedule_once(add_next, 0)
        self._outfits = list(outfits)
        self.has_outfits = bool(outfits)
        if outfits:
            deck.index = 0
        self.on_deck_index()

    def _new_card(self) -> OutfitCard:
        card = OutfitCard()
        card.bind(on_wear=self._wear, on_save=self._save)
        return card

    def _make_card(self, outfit: Dict[str, Any], saved: bool, worn: bool = False) -> OutfitCard:
        return self._fill_card(self._new_card(), outfit, saved, worn)

    def _fill_card(self, card: OutfitCard, outfit: Dict[str, Any], saved: bool, worn: bool) -> OutfitCard:
        pieces = [self._items[i] for i in outfit["item_ids"] if i in self._items]
        total = max(0.0, min(1.0, float(outfit["scores"].get("total", 0))))
        tops = sum(1 for p in pieces if _slot(p) == "top")
        if outfit.get("source") == "shuffle":
            kicker = "VIRTUAL SHUFFLE"
        elif outfit.get("source") == "trend":
            kicker = "RECREATED TREND"
        else:
            kicker = f"{OCCASION_LABELS.get(outfit.get('occasion', ''), outfit.get('occasion', '')).upper()}" \
                     f" · {tops} LAYER{'S' if tops != 1 else ''}"
        card.title = outfit["title"]
        card.subtitle = " · ".join(item_label(p) for p in pieces)
        card.kicker = kicker
        card.score = total
        card.score_text = f"{int(round(total * 100))}"
        card.rule_short = RULE_SHORT.get(outfit.get("harmony_rule", ""), "")
        card.saved = saved
        card.worn = worn
        card.shuffle = outfit.get("source") == "shuffle"
        card.outfit = outfit
        # Lazy: garment images are composed only when the card (or a
        # neighbour) is about to be seen - see on_deck_index.
        card.pending_pieces = pieces
        box = card.ids.reasons
        pills = list(reversed(box.children))
        wanted = outfit.get("reasons", [])[1:4]
        for j, text in enumerate(wanted):
            if j < len(pills):
                pills[j].text = text
            else:
                box.add_widget(ReasonPill(text=text))
        for extra in pills[len(wanted):]:
            box.remove_widget(extra)
        return card

    def on_deck_index(self) -> None:
        deck = self.ids.deck
        dots = self.ids.dots
        dots.clear_widgets()
        count = len(deck.slides)
        current = deck.index or 0
        app = self._app()
        for i in range(min(count, 12)):
            dot = ColorDot(rgba=list(app.c_text) if i == current else list(app.c_border))
            dot.size = (dp(7), dp(7)) if i != current else (dp(18), dp(7))
            dots.add_widget(dot)
        storage = app.storage
        slides = deck.slides
        for i in range(max(0, current - 1), min(len(slides), current + 2)):
            card = slides[i]
            if isinstance(card, OutfitCard) and card.pending_pieces is not None:
                card.ids.flatlay.compose(card.pending_pieces, storage)
                card.pending_pieces = None
        slide = deck.current_slide
        if isinstance(slide, OutfitCard):
            slide.ids.flatlay.play_intro()

    def _wear(self, card: OutfitCard) -> None:
        if card.worn:
            self.ids.toast.flash("Already logged for today", icon="check-circle")
            return
        try:
            updated = log_wear(self._app().storage, card.outfit)
        except StorageError as exc:
            self._app().show_dialog("Could not log", str(exc))
            return
        card.worn = True
        self.ids.toast.flash(f"Worn today · {updated} piece{'s' if updated != 1 else ''} updated",
                             icon="check-circle")

    def _save(self, card: OutfitCard) -> None:
        """Instant heart toggle (atomic write to outfits.json)."""
        outfit = dict(card.outfit)
        now_saved = not card.saved

        def _apply(doc: Dict[str, Any]) -> None:
            outfits = doc.setdefault("outfits", {})
            if now_saved:
                stored = outfits.setdefault(outfit["outfit_id"], outfit)
                stored["saved"] = True
            elif outfit["outfit_id"] in outfits:
                stored = outfits[outfit["outfit_id"]]
                if stored.get("wear_count"):
                    stored["saved"] = False  # keep its wear history
                else:
                    del outfits[outfit["outfit_id"]]

        card.saved = now_saved  # instant visual feedback; persisted right after
        card.pop_heart()
        try:
            self._app().storage.update("outfits", _apply)
        except StorageError as exc:
            card.saved = not now_saved
            self._app().show_dialog("Could not save", str(exc))
            return
        self.ids.toast.flash("Saved to your looks" if now_saved else "Removed from saved",
                             icon="heart" if now_saved else "heart-outline")

    # -- weather --------------------------------------------------------------
    def _apply_weather(self, weather: Optional[WeatherContext], message: str) -> None:
        self._weather = weather
        if weather is not None:
            place = weather.place.split(",")[0] if weather.place else ""
            self.weather_text = f"{round(weather.temp)}° feels · {weather.text}" + (f" · {place}" if place else "")
            self.weather_icon = weather.icon
            self.weather_live = weather.source in ("live", "cache")
        elif not self.weather_on:
            self.weather_text = "Weather off · tap to set up"
            self.weather_icon = "cloud-off-outline"
            self.weather_live = False
        else:
            self.weather_text = message or "Set your city below"
            self.weather_icon = "map-marker-question-outline"
            self.weather_live = False

    def refresh_weather(self) -> None:
        if not self.weather_on:
            self.ids.toast.flash("Turn on Weather in Live data below", icon="cloud-off-outline")
            return
        self._fetch_weather(force=True)

    def _fetch_weather(self, force: bool = False) -> None:
        st = self._app().storage
        if force:
            try:
                os.remove(st.path(WEATHER_CACHE_REL))
            except OSError:
                pass

        def done(result: Any) -> None:
            if isinstance(result, Exception):
                self._apply_weather(None, str(result))
                return
            weather, message = result
            changed = (weather is None) != (self._weather is None) or (
                weather is not None and self._weather is not None
                and (round(weather.temp) != round(self._weather.temp) or weather.wet != self._weather.wet))
            self._apply_weather(weather, message)
            if changed:
                self.generate()

        self._run("weather", lambda: load_weather(st, allow_network=True), done)

    def toggle_weather(self) -> None:
        st = self._app().storage
        self.weather_on = not self.weather_on
        st.set_setting("network.weather_enabled", self.weather_on)
        if not self.weather_on:
            self._apply_weather(None, "")
            self.generate()
            return
        loc = st.load("profile").get("location") or {}
        if loc.get("lat") is None:
            self._apply_weather(None, "Set your city below")
            self.ids.toast.flash("Now enter your city and tap Set city", icon="map-marker-outline")
        else:
            self._fetch_weather(force=True)

    def set_city(self) -> None:
        name = self.ids.city_field.text.strip()
        if not name:
            return
        if not self.weather_on:
            self._app().show_dialog("Weather is off", "Turn Weather on first. Setting the city sends its name "
                                                      "to Open-Meteo's free geocoding service to find it.")
            return
        st = self._app().storage

        def work() -> Dict[str, Any]:
            allowed, why = network_allowed(st)
            if not allowed:
                raise NetworkError(why)
            loc = geocode_city(name)
            st.update("profile", lambda d: d.setdefault("location", {}).update(
                {"lat": loc["lat"], "lon": loc["lon"], "city": loc["name"], "manual": True}))
            return loc

        def done(result: Any) -> None:
            if isinstance(result, Exception):
                self._app().show_dialog("City not set", str(result))
                return
            self.ids.toast.flash(f"Weather for {result['name'].split(',')[0]}", icon="map-marker-check-outline")
            self._fetch_weather(force=True)

        self._run("city", work, done)

    # -- trends + gaps ----------------------------------------------------------
    def toggle_trends(self) -> None:
        self.trends_on = not self.trends_on
        self._app().storage.set_setting("network.trends_enabled", self.trends_on)
        self.render_trends()
        if self.trends_on and (self._app().storage.load("trends").get("sources") or []):
            self._sync_trends(force=True)

    def add_feed(self) -> None:
        url = self.ids.feed_field.text.strip()
        if not url.startswith(("https://", "http://")):
            self._app().show_dialog("Add feed", "Enter a full RSS or Atom feed address starting with https://")
            return
        st = self._app().storage

        def _apply(doc: Dict[str, Any]) -> None:
            sources = doc.setdefault("sources", [])
            if not any(s.get("url") == url for s in sources):
                sources.append({"id": _outfit_id([url]), "url": url, "type": "feed", "enabled": True,
                                "last_status": "not synced yet", "etag": "", "last_modified": ""})

        st.update("trends", _apply)
        self.ids.feed_field.text = ""
        self.render_feeds()
        if self.trends_on:
            self._sync_trends(force=True)
        else:
            self.ids.toast.flash("Feed added · turn Trend feeds on to sync", icon="rss")

    def remove_feed(self, url: str) -> None:
        def _apply(doc: Dict[str, Any]) -> None:
            doc["sources"] = [s for s in doc.get("sources") or [] if s.get("url") != url]
            doc["trends"] = build_trends(doc["sources"])
            items = list(self._items.values())
            doc["matches"] = match_trends(items, doc["trends"])

        st = self._app().storage
        doc = st.update("trends", _apply)
        refresh_wishlist(st, list(self._items.values()), doc)
        self.render_feeds()
        self.render_trends()

    def sync_trends_now(self) -> None:
        if not self.trends_on:
            self.ids.toast.flash("Turn on Trend feeds in Live data below", icon="rss")
            return
        self._sync_trends(force=True)

    def _sync_trends(self, force: bool, quiet: bool = False) -> None:
        st = self._app().storage

        def done(result: Any) -> None:
            if isinstance(result, Exception):
                self._app().show_dialog("Trend sync failed", str(result))
                return
            _doc, message = result
            self.render_trends()
            self.render_feeds()
            if message and not quiet:
                self._app().show_dialog("Trend sync", message)
            elif not quiet:
                self.ids.toast.flash("Trends synced", icon="sync")

        self.trend_caption = "Syncing feeds…"
        self._run("trends", lambda: sync_trends(st, force=force), done)

    def render_trends(self) -> None:
        st = self._app().storage
        doc = st.load("trends")
        trends = doc.get("trends") or []
        matches = {m["trend_id"]: m for m in doc.get("matches") or []}
        box = self.ids.trend_list
        box.clear_widgets()
        if not self.trends_on:
            self.trend_caption = "Off. Turn on Trend feeds in Live data to compare your wardrobe with real looks."
        elif not doc.get("sources"):
            self.trend_caption = "Add an RSS or Atom feed from a fashion site or lookbook below."
        elif not trends:
            self.trend_caption = "No colour + garment combinations found in your feeds yet."
        else:
            owned = sum(1 for t in trends if matches.get(t["trend_id"], {}).get("match_pct") == 100)
            last = doc.get("last_sync")
            ago = f" · synced {int((time.time() - float(last)) / 3600)} h ago" if isinstance(last, (int, float)) else ""
            self.trend_caption = f"{len(trends)} trending looks · you own {owned} completely{ago}"
        wish_keys = {(w.get("category"), w.get("color_name"))
                     for w in st.load("wishlist").get("wishes") or [] if not w.get("acquired")}
        shown = trends[:10] if self.trends_on else []
        for trend in shown:
            match = matches.get(trend["trend_id"], {})
            box.add_widget(self._make_look_card(trend, match, wish_keys))
        self.has_trends = bool(shown)
        self.ids.trend_scroll.scroll_x = 0

        wishes = [w for w in st.load("wishlist").get("wishes") or [] if not w.get("acquired")]
        wishes.sort(key=lambda w: (w.get("unlocks_outfits", 0), w.get("priority", 0)), reverse=True)
        gaps = self.ids.gap_list
        gaps.clear_widgets()
        if wishes:
            self.gap_caption = "One piece away from a trending look - ranked by how many new outfits each unlocks."
        else:
            self.gap_caption = ("No gaps found - your wardrobe covers the synced trends."
                                if trends else "Gaps appear after a trend sync.")
        for wish in wishes[:8]:
            row = ListRow(
                title=f"{wish.get('color_name', '')} {wish.get('category', '')}",
                subtitle=f"Unlocks {wish.get('unlocks_outfits', 0)} outfits · {wish.get('trend_title', '')}",
                action_icon="close",
                payload=wish.get("wish_id"),
            )
            row.set_swatches([wish.get("hex_target", "#808080")])
            row.bind(on_action=lambda r: self.dismiss_gap(r.payload))
            gaps.add_widget(row)

    # -- trend discovery cards ---------------------------------------------------
    def _look_parts(self, trend: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Each trend component with the owned item that fills it (or None)."""
        active = [i for i in self._items.values() if i.get("status", "active") == "active"]
        parts = []
        for comp in trend.get("components", []):
            owner = next((i for i in active if item_matches_component(i, comp)), None)
            parts.append({"component": comp, "piece": owner, "hex": comp.get("hex_target", "#808080"),
                          "category": comp.get("category", "")})
        return parts

    def _make_look_card(self, trend: Dict[str, Any], match: Dict[str, Any],
                        wish_keys: set) -> LookCard:
        parts = self._look_parts(trend)
        missing = [p for p in parts if p["piece"] is None]
        if not missing:
            cta, active = "100% OWNED · RECREATE LOOK", True
        elif len(missing) == 1:
            comp = missing[0]["component"]
            on_list = (comp.get("category"), comp.get("color_name")) in wish_keys
            cta = "MISSING 1 PIECE · ON WISHLIST" if on_list else "MISSING 1 PIECE · ADD TO WISHLIST"
            active = False
        else:
            cta, active = f"MISSING {len(missing)} PIECES · VIEW LOOK", False
        card = LookCard(
            title=" + ".join(f"{c['color_name']} {c['category']}" for c in trend.get("components", [])),
            subtitle=trend.get("title", ""),
            cta=cta,
            cta_active=active,
        )
        card.trend, card.match = trend, match
        card.ids.collage.compose(parts, self._app().storage)
        card.set_dots(trend.get("palette") or [])
        card.bind(on_chip=self._on_look_cta, on_release=lambda c: self.open_look(c.trend))
        return card

    def _on_look_cta(self, card: LookCard) -> None:
        parts = self._look_parts(card.trend)
        missing = [p for p in parts if p["piece"] is None]
        if not missing:
            self.recreate_look(card.trend)
        elif len(missing) == 1:
            self.add_to_wishlist(card.trend, [missing[0]["component"]])
        else:
            self.open_look(card.trend)

    def recreate_look(self, trend: Dict[str, Any]) -> None:
        """Build an outfit from the owned pieces of a trend and put it on top of the deck."""
        pieces = [p["piece"] for p in self._look_parts(trend) if p["piece"] is not None]
        ctx = self._ctx()
        outfit = None
        for occasion in [self.occasion] + [o for o in OCCASIONS if o != self.occasion]:
            ctx.occasion, ctx.weather = occasion, None
            found = generate_outfits(pieces, ctx, limit=1, diversify=False)
            if found:
                outfit = found[0]
                break
        if outfit is None:  # e.g. the look has no top + bottom pair: physics-only fallback
            found = generate_outfits(pieces, ctx, limit=1, shuffle=True, seed=1, diversify=False)
            outfit = found[0] if found else None
        if outfit is None:
            self._app().show_dialog("Recreate look", "This look needs at least a top and a bottom you own.")
            return
        outfit["source"] = "trend"
        outfit["reasons"] = [outfit["reasons"][0] if outfit["reasons"] else "", f"Trend: {trend.get('title', '')}"] \
            + outfit["reasons"][1:3]
        self._render_deck([outfit] + [o for o in self._outfits if o["outfit_id"] != outfit["outfit_id"]])
        self.children[-1].scroll_y = 1.0  # back to the deck
        self.ids.toast.flash("Look recreated from your wardrobe", icon="star-four-points")

    def add_to_wishlist(self, trend: Dict[str, Any], components: List[Dict[str, Any]]) -> None:
        st = self._app().storage
        active = [i for i in self._items.values() if i.get("status", "active") == "active"]

        def _apply(doc: Dict[str, Any]) -> None:
            wishes = doc.setdefault("wishes", [])
            dismissed = doc.setdefault("dismissed", [])
            for comp in components:
                key = f"{comp['category']}|{comp['color_name']}"
                wish_id = _outfit_id([key])
                if wish_id in dismissed:
                    dismissed.remove(wish_id)
                if any(w.get("wish_id") == wish_id for w in wishes):
                    continue
                partner = {"top": {"bottom"}, "bottom": {"top"}}.get(comp.get("slot", ""), {"top", "bottom"})
                target = item_color({"colors": [], "primary_hex": comp["hex_target"], "color_family":
                                     "neutral" if comp["color_name"] in NEUTRAL_COLOR_NAMES else "cool"})
                unlocks = sum(1 for i in active if _slot(i) in partner
                              and harmony_score([item_color(i), target])[0] >= 0.6)
                wishes.append({"wish_id": wish_id, "category": comp["category"], "color_name": comp["color_name"],
                               "hex_target": comp["hex_target"], "reason": "trend_gap",
                               "trend_id": trend.get("trend_id"), "trend_title": trend.get("title", ""),
                               "unlocks_outfits": unlocks, "priority": trend.get("popularity_score", 0),
                               "added_at": utc_now_iso(), "acquired": False})

        st.update("wishlist", _apply)
        self.render_trends()
        names = ", ".join(f"{c['color_name']} {c['category']}" for c in components)
        self.ids.toast.flash(f"Added to wishlist · {names}", icon="heart-plus-outline")

    def open_look(self, trend: Dict[str, Any]) -> None:
        parts = self._look_parts(trend)
        missing = [p["component"] for p in parts if p["piece"] is None]
        owned = len(parts) - len(missing)
        pct = int(round(100.0 * owned / max(1, len(parts))))
        sheet = LookDetail(
            kicker=f"TREND LOOK · {pct}% OWNED",
            title=" + ".join(f"{p['component']['color_name']} {p['category']}" for p in parts),
            subtitle=trend.get("title", ""),
            primary_text="Recreate look" if not missing else "Add to wishlist",
            primary_icon="star-four-points" if not missing else "heart-plus-outline",
        )
        for part in parts:
            comp, piece = part["component"], part["piece"]
            row = ListRow(
                title=f"{comp['color_name']} {comp['category']}",
                subtitle=f"In your wardrobe · {item_label(piece)}" if piece else "Not in your wardrobe yet",
                badge="OWNED" if piece else "MISSING",
                badge_on=piece is not None,
            )
            row.set_swatches([str(piece.get("primary_hex")) if piece else comp["hex_target"]])
            sheet.ids.rows.add_widget(row)

        def primary(_sheet: Any) -> None:
            sheet.dismiss()
            if missing:
                self.add_to_wishlist(trend, missing)
            else:
                self.recreate_look(trend)

        sheet.bind(on_primary=primary)
        sheet.open()

    def dismiss_gap(self, wish_id: str) -> None:
        def _apply(doc: Dict[str, Any]) -> None:
            doc["wishes"] = [w for w in doc.get("wishes") or [] if w.get("wish_id") != wish_id]
            dismissed = doc.setdefault("dismissed", [])
            if wish_id not in dismissed:
                dismissed.append(wish_id)

        self._app().storage.update("wishlist", _apply)
        self.render_trends()

    def render_feeds(self) -> None:
        box = self.ids.feed_list
        box.clear_widgets()
        for src in self._app().storage.load("trends").get("sources") or []:
            url = str(src.get("url", ""))
            host = url.split("://", 1)[-1].split("/", 1)[0]
            row = ListRow(title=host or url, subtitle=str(src.get("last_status", "")), action_icon="close", payload=url)
            row.bind(on_action=lambda r: self.remove_feed(r.payload))
            box.add_widget(row)

    # -- packing wizard -----------------------------------------------------------
    def change_days(self, delta: int) -> None:
        self.pack_days = int(max(1, min(21, self.pack_days + delta)))

    def _render_pack_occasions(self) -> None:
        box = self.ids.pack_occasions
        if len(box.children) != len(OCCASIONS):
            box.clear_widgets()
            for occ in OCCASIONS:
                chip = ChipButton(text=OCCASION_LABELS.get(occ, occ), value=occ)
                chip.bind(on_release=lambda btn: self._toggle_pack_occasion(btn.value))
                box.add_widget(chip)
        for chip in box.children:
            chip.selected = chip.value in self._pack_occasions
            chip.icon = "check" if chip.selected else ""

    def _toggle_pack_occasion(self, occ: str) -> None:
        if occ in self._pack_occasions and len(self._pack_occasions) > 1:
            self._pack_occasions.discard(occ)
        else:
            self._pack_occasions.add(occ)
        self._render_pack_occasions()

    def build_capsule(self) -> None:
        from datetime import date, timedelta
        try:
            start = date.fromisoformat(self.ids.pack_start.text.strip())
        except ValueError:
            self._app().show_dialog("Packing wizard", "Enter the start date as YYYY-MM-DD.")
            return
        days = int(self.pack_days)
        dest = self.ids.pack_dest.text.strip()
        occasions = [o for o in OCCASIONS if o in self._pack_occasions]
        st = self._app().storage
        items = list(self._items.values())
        ctx = self._ctx()
        use_network = self.weather_on and bool(dest)

        def work() -> Dict[str, Any]:
            dates = [(start + timedelta(days=i)).isoformat() for i in range(days)]
            notes: List[str] = []
            place, forecast_days = dest, {}
            if use_network:
                allowed, why = network_allowed(st)
                if allowed:
                    try:
                        loc = geocode_city(dest)
                        place = loc["name"]
                        fc = fetch_forecast(loc["lat"], loc["lon"], days=16)
                        forecast_days = {d["date"]: d for d in fc.get("daily") or []}
                    except NetworkError as exc:
                        notes.append(f"Forecast unavailable ({exc}).")
                else:
                    notes.append(why)
            elif dest and not self.weather_on:
                notes.append("Weather is off - planned for mild weather.")
            day_weather = [weather_from_day(forecast_days[d], place) if d in forecast_days
                           else assumed_weather(d, place) for d in dates]
            missing = sum(1 for d in dates if d not in forecast_days)
            if forecast_days and missing:
                notes.append(f"{missing} day(s) are beyond the 16-day forecast - assumed mild.")
            plan = plan_capsule(items, day_weather, occasions, palette=ctx.palette,
                                undertone_bias=ctx.undertone_bias, body_shape=ctx.body_shape)
            plan["notes"] = notes + plan["notes"]
            trip = {"trip_id": _outfit_id([dest, start.isoformat(), str(days), utc_now_iso()]),
                    "destination": place or "Trip", "start": dates[0], "end": dates[-1],
                    "forecast_summary": [{"date": w.date, "temp": round(w.temp), "text": w.text,
                                          "source": w.source} for w in day_weather],
                    "occasions_needed": occasions, "capsule_item_ids": plan["capsule_item_ids"],
                    "outfit_plan": [{"day": p["day"], "date": p["date"],
                                     "outfit_id": p["outfit"]["outfit_id"] if p["outfit"] else None,
                                     "item_ids": p["outfit"]["item_ids"] if p["outfit"] else []}
                                    for p in plan["outfit_plan"]],
                    "variation_count": plan["variation_count"], "created_at": utc_now_iso()}
            st.update("trips", lambda d: d.setdefault("trips", []).append(trip))
            return {"trip": trip, "plan": plan}

        def done(result: Any) -> None:
            if isinstance(result, Exception):
                self._app().show_dialog("Packing wizard", f"Could not build the capsule: {result}")
                return
            self._render_capsule(result["trip"], result["plan"])

        self.pack_summary = "Planning your capsule…"
        self._run("capsule", work, done)

    def _render_capsule(self, trip: Dict[str, Any], plan: Dict[str, Any]) -> None:
        st = self._app().storage
        n_items = len(trip["capsule_item_ids"])
        days = len(plan["outfit_plan"])
        if not n_items:
            self.pack_summary = "Not enough pieces for this trip yet - scan a few tops and bottoms."
        else:
            self.pack_summary = (f"{trip['destination']}: pack {n_items} pieces -> "
                                 f"{plan['variation_count']} outfit combinations for {days} days."
                                 + ("  " + " ".join(plan["notes"]) if plan["notes"] else ""))
        items_box = self.ids.pack_items
        items_box.clear_widgets()
        for iid in trip["capsule_item_ids"]:
            piece = self._items.get(iid)
            if piece is None:
                continue
            thumb_rel = (piece.get("images") or {}).get("thumb")
            thumb = st.path(thumb_rel) if thumb_rel else ""
            items_box.add_widget(ItemThumb(source=thumb if thumb and os.path.isfile(thumb) else "",
                                           caption=str(piece.get("category", "")),
                                           layer_text=str(piece.get("color_family", "")).upper(),
                                           swatch=str(piece.get("primary_hex") or "")))
        plan_box = self.ids.pack_plan
        plan_box.clear_widgets()
        for day in plan["outfit_plan"]:
            outfit = day["outfit"]
            pieces = [self._items[i] for i in (outfit or {}).get("item_ids", []) if i in self._items]
            w = day["weather"]
            row = ListRow(
                title=f"Day {day['day']} · {_short_date(day['date'])} · {w['temp']}° {w['text']}",
                subtitle=" + ".join(item_label(p) for p in pieces) if pieces else "No complete outfit for this weather",
            )
            row.set_swatches([str(p.get("primary_hex") or "#808080") for p in pieces])
            plan_box.add_widget(row)


class MetricTile(MDCard):
    """Dashboard tile: icon (or colour swatch), label, big value, caption."""

    icon = StringProperty("help-circle")
    label = StringProperty("")
    value = StringProperty("—")
    caption = StringProperty("")
    swatch = StringProperty("")


class BodyPhotoFrame(MDCard):
    """Photo container for the profile picture plus the landmark overlay.

    Dispatches ``on_point_tapped(u, v)`` with normalised image coordinates
    (top-left origin) when the user taps the photo in manual mode.
    """

    photo_source = StringProperty("")
    points = DictProperty({})
    face_box = ListProperty([])
    skin_point = ListProperty([])
    manual_mode = BooleanProperty(False)
    pill_text = StringProperty("")

    __events__ = ("on_point_tapped",)

    def on_point_tapped(self, u: float, v: float) -> None:
        pass


LANDMARK_WAIST_RGBA = get_color_from_hex("#F59E0B")
LANDMARK_HIP_RGBA = get_color_from_hex("#EC4899")


class LandmarkOverlay(Widget):
    """Draws detected / tapped landmarks over the photo and, in manual mode,
    converts taps into normalised image coordinates."""

    frame = ObjectProperty(None, allownone=True)
    image = ObjectProperty(None, allownone=True)
    points = DictProperty({})
    face_box = ListProperty([])
    skin_point = ListProperty([])
    manual_mode = BooleanProperty(False)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trigger = Clock.create_trigger(self.redraw, 0)
        self.bind(pos=self._trigger, size=self._trigger, points=self._trigger,
                  face_box=self._trigger, skin_point=self._trigger, manual_mode=self._trigger)
        app = MDApp.get_running_app()
        if app is not None:
            app.bind(c_accent=self._trigger, c_text=self._trigger)

    def on_image(self, _inst: Any, img: Any) -> None:
        if img is not None:
            img.bind(texture=self._trigger, pos=self._trigger, size=self._trigger,
                     norm_image_size=self._trigger)
        self._trigger()

    def image_rect(self) -> Optional[Tuple[float, float, float, float]]:
        """(x, y, w, h) of the displayed picture inside the 'contain' Image."""
        img = self.image
        if img is None or img.texture is None or not img.source:
            return None
        dw, dh = img.norm_image_size
        if dw <= 1 or dh <= 1:
            return None
        return img.center_x - dw / 2.0, img.center_y - dh / 2.0, float(dw), float(dh)

    def redraw(self, *_args: Any) -> None:
        self.canvas.after.clear()
        rect = self.image_rect()
        if rect is None:
            return
        x0, y0, dw, dh = rect
        app = MDApp.get_running_app()
        accent = list(app.c_accent) if app is not None else [0.0, 0.9, 0.8, 1.0]
        ink = list(app.c_text) if app is not None else [1.0, 1.0, 1.0, 1.0]

        def xy(p: List[float]) -> Tuple[float, float]:
            return x0 + p[0] * dw, y0 + (1.0 - p[1]) * dh

        with self.canvas.after:
            if len(self.face_box) == 4 and not self.manual_mode:
                u, v, bw, bh = self.face_box
                Color(accent[0], accent[1], accent[2], 0.65)
                Line(rounded_rectangle=(x0 + u * dw, y0 + (1.0 - v - bh) * dh, bw * dw, bh * dh, dp(8)),
                     width=dp(1.2))
            for name, col in (("shoulder", accent), ("waist", LANDMARK_WAIST_RGBA),
                              ("hip", LANDMARK_HIP_RGBA)):
                left, right = self.points.get(f"{name}_l"), self.points.get(f"{name}_r")
                if left and right:
                    Color(col[0], col[1], col[2], 0.9)
                    Line(points=[*xy(left), *xy(right)], width=dp(1.6))
                for p in (left, right):
                    if p:
                        px, py = xy(p)
                        Color(1, 1, 1, 1)
                        Ellipse(pos=(px - dp(7), py - dp(7)), size=(dp(14), dp(14)))
                        Color(col[0], col[1], col[2], 1)
                        Ellipse(pos=(px - dp(5), py - dp(5)), size=(dp(10), dp(10)))
            if len(self.skin_point) == 2:
                px, py = xy(self.skin_point)
                Color(1, 1, 1, 1)
                Line(circle=(px, py, dp(10)), width=dp(2))
                Color(ink[0], ink[1], ink[2], 1)
                Line(circle=(px, py, dp(6)), width=dp(1.2))

    def on_touch_down(self, touch: Any) -> bool:
        if not self.manual_mode or not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        rect = self.image_rect()
        if rect is None:
            return True
        x0, y0, dw, dh = rect
        u = (touch.x - x0) / dw
        v = 1.0 - (touch.y - y0) / dh
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 and self.frame is not None:
            self.frame.dispatch("on_point_tapped", round(u, 4), round(v, 4))
        return True


MANUAL_BODY_STEPS: List[Tuple[str, str]] = [
    ("shoulder_l", "the outer edge of your LEFT shoulder"),
    ("shoulder_r", "the outer edge of your RIGHT shoulder"),
    ("waist_l", "the LEFT side of your waist (narrowest point)"),
    ("waist_r", "the RIGHT side of your waist"),
    ("hip_l", "the LEFT side of your hips (widest point)"),
    ("hip_r", "the RIGHT side of your hips"),
]
MANUAL_SKIN_STEP: Tuple[str, str] = ("skin", "your cheek - bare, evenly lit skin")

BODY_METHOD_LABELS: Dict[str, str] = {
    "auto_face_scaled": "Auto · face-scaled",
    "auto_height_scaled": "Auto · height-scaled",
    "manual_tap": "Tapped by you",
}


class ProfileZone(MDFloatLayout):
    """Profile tab: biometric dashboard, manual tap override, cut-out engine,
    storage diagnostics."""

    status_text = StringProperty("Add a full-body photo to measure your proportions and skin tone.")
    photo_source = StringProperty("")
    has_photo = BooleanProperty(False)
    points = DictProperty({})
    face_box = ListProperty([])
    skin_point = ListProperty([])
    method_pill = StringProperty("")

    manual_mode = BooleanProperty(False)
    manual_prompt = StringProperty("")
    manual_step_text = StringProperty("")

    shape_value = StringProperty("—")
    shape_caption = StringProperty("Add a photo")
    sw_value = StringProperty("—")
    sw_caption = StringProperty("Add a photo")
    wh_value = StringProperty("—")
    wh_caption = StringProperty("Add a photo")
    tone_value = StringProperty("—")
    tone_caption = StringProperty("Add a photo")
    tone_swatch = StringProperty("")
    depth_value = StringProperty("—")
    depth_caption = StringProperty("Add a photo")
    season_value = StringProperty("—")
    season_caption = StringProperty("Add a photo")
    palette_caption = StringProperty("Appears once your skin tone is measured.")

    MIN_OVERLAY_SECONDS = 0.7

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._busy = False
        self._overlay: Optional[ScanOverlay] = None
        self._steps: List[Tuple[str, str]] = []
        self._step = 0
        self._backup: Optional[Tuple[Dict[str, Any], List[float]]] = None
        self._manual_points: Dict[str, List[float]] = {}
        self._manual_skin: Optional[List[float]] = None
        self._session_photo = ""
        self._image_size: Optional[Tuple[int, int]] = None

    def on_kv_post(self, base_widget: Any) -> None:
        Clock.schedule_once(lambda _dt: self.refresh(), 0)

    @staticmethod
    def _app() -> "WardrobeApp":
        return MDApp.get_running_app()  # type: ignore[return-value]

    def refresh(self) -> None:
        self.refresh_storage_status()
        self.load_profile()

    # -- rendering ----------------------------------------------------------
    def load_profile(self) -> None:
        app = self._app()
        if app is None or app.storage is None:
            return
        self._render(app.storage.load("profile"))

    def _render(self, doc: Dict[str, Any]) -> None:
        app = self._app()
        body = doc.get("body") or {}
        skin = doc.get("skin") or {}
        palette = doc.get("palette") or {}
        analysis = doc.get("analysis") or {}

        path = ""
        rel = doc.get("source_photo")
        if rel:
            candidate = app.storage.path(rel)
            if os.path.isfile(candidate):
                path = candidate
        if not path and self._session_photo and os.path.isfile(self._session_photo):
            path = self._session_photo
        self.photo_source = path
        self.has_photo = bool(path)
        size = analysis.get("image_size") or body.get("image_size")
        self._image_size = (int(size[0]), int(size[1])) if size else None

        if not self.manual_mode:
            self.points = dict(body.get("points") or {})
            self.face_box = list(analysis.get("face_box") or [])
            self.skin_point = list(skin.get("point") or [])
            method = body.get("detection_method", "")
            self.method_pill = BODY_METHOD_LABELS.get(method, "Processed on-device") if body else (
                "Processed on-device" if path else "")

        # Body tiles
        shape = body.get("shape")
        conf = body.get("confidence")
        conf_txt = f" · {int(round(conf * 100))}% sure" if isinstance(conf, (int, float)) else ""
        if body:
            self.shape_value = BODY_SHAPE_LABELS.get(shape, "Partial")
            self.shape_caption = BODY_SHAPE_TIPS.get(shape, "Needs shoulders, waist and hips in view.")
            sw = body.get("shoulder_to_waist_ratio")
            self.sw_value = f"{sw:.2f}" if sw else "—"
            self.sw_caption = BODY_METHOD_LABELS.get(body.get("detection_method", ""), "Measured") + conf_txt
            wh = body.get("waist_to_hip_ratio")
            self.wh_value = f"{wh:.2f}" if wh else "—"
            self.wh_caption = "Front-view silhouette ratio" if wh else "Hips not in view"
        else:
            for prop in ("shape", "sw", "wh"):
                setattr(self, f"{prop}_value", "—")
                setattr(self, f"{prop}_caption", "Add a photo" if not path else "Tap Adjust points")

        # Skin tiles
        if skin:
            self.tone_value = str(skin.get("undertone", "—"))
            bias = skin.get("undertone_bias", "")
            self.tone_caption = f"{skin.get('hue_angle', 0):.0f}° hue · {bias} bias"
            self.tone_swatch = str(skin.get("sample_hex", ""))
            self.depth_value = str(skin.get("depth", "—")).capitalize()
            self.depth_caption = f"ITA {skin.get('ita_angle', 0):.0f}°" + (
                " · light-corrected" if skin.get("white_balance") == "backdrop" else "")
            self.season_value = str(skin.get("seasonal_palette", "—"))
            self.season_caption = "Tapped sample" if skin.get("detection_method") == "manual_tap" else "From your cheeks"
        else:
            self.tone_value = self.depth_value = self.season_value = "—"
            self.tone_swatch = ""
            hint = "Add a photo" if not path else "Tap Adjust points"
            self.tone_caption = self.depth_caption = self.season_caption = hint

        # Palette
        self._fill_swatches("best_colors", [(c["name"], c["hex"]) for c in palette.get("flattering_colors", [])])
        self._fill_swatches("best_neutrals", list(zip(palette.get("best_neutral_names", []),
                                                     palette.get("best_neutrals", []))))
        self._fill_swatches("avoid_colors", [(c["name"], c["hex"]) for c in palette.get("avoid_colors", [])])
        self.palette_caption = (f"General colour-theory guide for a {palette['basis']}."
                                if palette.get("basis") else "Appears once your skin tone is measured.")

        if not self.manual_mode:
            if body or skin:
                when = str(analysis.get("manual_adjusted_at") or analysis.get("analyzed_at") or "")[:10]
                self.status_text = f"Saved privately on this device{' · ' + when if when else ''}."
                notes = (body.get("notes") or []) + [n for n in analysis.get("notes", []) if "tap" in n.lower()]
                if notes and body.get("detection_method") != "manual_tap":
                    self.status_text += " " + notes[0]
            else:
                self.status_text = "Add a full-body photo to measure your proportions and skin tone."

    def _fill_swatches(self, container_id: str, entries: List[Tuple[str, str]]) -> None:
        box = self.ids[container_id]
        box.clear_widgets()
        for name, hex_code in entries:
            box.add_widget(SwatchPill(hex_color=hex_code, label=name, detail=""))

    # -- capture --------------------------------------------------------------
    def _ready(self) -> bool:
        app = self._app()
        if CV_AVAILABLE and app.profiler is not None:
            return True
        app.show_dialog("Vision engine missing",
                        "OpenCV and NumPy are required for profile analysis.\n\n"
                        "Desktop: pip install -r requirements.txt")
        return False

    def open_camera(self) -> None:
        if self._busy or not self._ready():
            return
        self._app().open_camera_for("profile")

    def open_gallery(self) -> None:
        if self._busy or not self._ready():
            return
        self._app().pick_image(self.start_analysis)

    def start_analysis(self, path: str, cleanup: bool = False, exif_orientation: int = 1) -> None:
        if self._busy:
            return
        if self.manual_mode:
            self.cancel_manual()
        self._busy = True
        if self._overlay is None:
            self._overlay = ScanOverlay()
        self._overlay.show(path, "Body photos never leave this device", title="Analysing Body Profile")
        threading.Thread(target=self._worker, args=(path, cleanup, exif_orientation), daemon=True).start()

    def _post_progress(self, text: str) -> None:
        def _apply(_dt: float) -> None:
            if self._overlay is not None:
                self._overlay.set_stage(text)

        Clock.schedule_once(_apply, 0)

    def _worker(self, path: str, cleanup: bool, exif_orientation: int) -> None:
        app = self._app()
        try:
            scan = app.profiler.analyze(path, exif_orientation, self._post_progress)
            jpeg = encode_image(".jpg", scan.photo_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            Clock.schedule_once(lambda _dt: self._finish(lambda: self._on_scan(scan, jpeg), scan.processing_ms), 0)
        except VisionError as exc:
            message = str(exc)
            Clock.schedule_once(lambda _dt: self._finish(lambda: app.show_dialog("Profile scan failed", message)), 0)
        except Exception as exc:  # noqa: BLE001 - never let the worker die silently
            Logger.exception(f"{LOG_TAG}: profile scan failed")
            message = f"Unexpected error while analysing the photo: {exc}"
            Clock.schedule_once(lambda _dt: self._finish(lambda: app.show_dialog("Profile scan failed", message)), 0)
        finally:
            if cleanup:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _finish(self, then: Callable[[], None], processing_ms: Optional[int] = None) -> None:
        overlay = self._overlay
        elapsed = time.perf_counter() - overlay.opened_at if overlay is not None else 1e9
        delay = max(0.0, self.MIN_OVERLAY_SECONDS - elapsed)
        if processing_ms is not None and overlay is not None:
            overlay.complete(processing_ms)
            delay = max(delay, 0.45)  # let "Done in 0.62 s" register

        def _close(_dt: float) -> None:
            if self._overlay is not None and self._overlay.is_showing:
                self._overlay.dismiss()
            self._busy = False
            then()

        Clock.schedule_once(_close, delay)

    def _on_scan(self, scan: BodyScan, jpeg: bytes) -> None:
        app = self._app()
        keep_photo = not bool(app.storage.get_setting("privacy.delete_source_photos", False))
        previous = app.storage.load("profile").get("source_photo")
        rel: Optional[str] = None
        try:
            if keep_photo:
                rel = f"profile/images/body_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
                atomic_write_bytes(app.storage.path(rel), jpeg)
                self._session_photo = ""
            else:
                # Privacy mode: keep the photo only for this session (purged later).
                self._session_photo = app.storage.path("cache", f"session_body_{uuid.uuid4().hex}.jpg")
                atomic_write_bytes(self._session_photo, jpeg)
        except (OSError, StorageError) as exc:
            app.show_dialog("Save failed", str(exc))
            return
        if previous and previous != rel:
            try:
                os.remove(app.storage.path(previous))
            except (OSError, StorageError):
                pass

        body = dict(scan.body) if scan.body else {}
        if body:
            body["image_size"] = list(scan.image_size)
        skin = dict(scan.skin) if scan.skin else {}
        palette = dict(scan.palette) if scan.palette else {}
        analysis = {
            "analyzed_at": utc_now_iso(),
            "engine_version": APP_VERSION,
            "image_size": list(scan.image_size),
            "face_box": scan.face_box_normalized(),
            "face_detector": scan.face_method,
            "segmenter": scan.segmenter,
            "processing_ms": scan.processing_ms,
            "notes": scan.notes,
        }

        def _apply(doc: Dict[str, Any]) -> None:
            doc["source_photo"] = rel
            doc["body"] = body
            doc["skin"] = skin
            doc["palette"] = palette
            doc["analysis"] = analysis

        try:
            doc = app.storage.update("profile", _apply)
        except StorageError as exc:
            app.show_dialog("Save failed", str(exc))
            return
        self._render(doc)

        if not body or not skin:
            steps: List[Tuple[str, str]] = []
            if not body:
                steps += MANUAL_BODY_STEPS
            if not skin:
                steps.append(MANUAL_SKIN_STEP)
            self.start_manual(steps)
            missing = " and ".join(x for x, ok in (("measurements", body), ("skin tone", skin)) if not ok)
            self.status_text = f"Couldn't detect your {missing} automatically - tap the points instead."
        else:
            self.ids.toast.flash(f"Profile saved · {BODY_SHAPE_LABELS.get(body.get('shape'), 'Measured')}")

    # -- manual tap override -------------------------------------------------
    def start_manual(self, steps: Optional[List[Tuple[str, str]]] = None) -> None:
        if not self.photo_source:
            self._app().show_dialog("Adjust points", "Add a photo first.")
            return
        if self._image_size is None:
            texture = self.ids.body_frame.ids.photo.texture
            if texture is not None:
                self._image_size = (int(texture.size[0]), int(texture.size[1]))
        self._steps = list(steps) if steps else list(MANUAL_BODY_STEPS)
        self._step = 0
        self._backup = (dict(self.points), list(self.skin_point))
        self._manual_points = {}
        self._manual_skin = None
        if any(key != "skin" for key, _ in self._steps):
            self.points = {}
        if any(key == "skin" for key, _ in self._steps):
            self.skin_point = []
        self.manual_mode = True
        self.method_pill = "Manual measure"
        self._update_prompt()

    def _update_prompt(self) -> None:
        _key, text = self._steps[self._step]
        self.manual_step_text = f"MANUAL MEASURE · STEP {self._step + 1} OF {len(self._steps)}"
        self.manual_prompt = f"Tap {text}"

    def add_manual_point(self, u: float, v: float) -> None:
        if not self.manual_mode or self._step >= len(self._steps):
            return
        key, _text = self._steps[self._step]
        if key == "skin":
            self._manual_skin = [u, v]
            self.skin_point = [u, v]
        else:
            self._manual_points[key] = [u, v]
            pts = dict(self.points)
            pts[key] = [u, v]
            self.points = pts
        self._step += 1
        if self._step >= len(self._steps):
            self._finish_manual()
        else:
            self._update_prompt()

    def undo_manual_point(self) -> None:
        if not self.manual_mode or self._step == 0:
            return
        self._step -= 1
        key, _text = self._steps[self._step]
        if key == "skin":
            self._manual_skin = None
            self.skin_point = []
        else:
            self._manual_points.pop(key, None)
            pts = dict(self.points)
            pts.pop(key, None)
            self.points = pts
        self._update_prompt()

    def cancel_manual(self) -> None:
        if not self.manual_mode:
            return
        self.manual_mode = False
        if self._backup is not None:
            self.points, self.skin_point = dict(self._backup[0]), list(self._backup[1])
        self.load_profile()

    def _finish_manual(self) -> None:
        app = self._app()
        self.manual_mode = False
        body_update: Optional[Dict[str, Any]] = None
        if self._manual_points and self._image_size:
            body_update = body_from_points(self._manual_points, self._image_size)
            if body_update is not None:
                body_update["image_size"] = list(self._image_size)

        skin_update: Optional[Dict[str, Any]] = None
        palette_update: Optional[Dict[str, Any]] = None
        if self._manual_skin is not None:
            raw = None
            try:
                bgr = limit_size(load_image_bgr(self.photo_source), PROFILE_WORK_SIDE)
                raw = sample_skin(bgr, point=(self._manual_skin[0], self._manual_skin[1]))
            except VisionError:
                raw = None
            if raw is not None:
                skin_update = skin_profile(raw, None, "manual_tap")
                skin_update["point"] = list(self._manual_skin)
                palette_update = build_palette(skin_update)
            else:
                app.show_dialog("Skin sample", "That spot could not be read as skin. "
                                               "Tap Adjust points and try a cheek area without shadow.")

        def _apply(doc: Dict[str, Any]) -> None:
            if body_update is not None:
                doc["body"] = body_update
            if skin_update is not None:
                doc["skin"] = skin_update
                doc["palette"] = palette_update
            analysis = doc.get("analysis")
            if not isinstance(analysis, dict):
                analysis = {}
                doc["analysis"] = analysis
            analysis["manual_adjusted_at"] = utc_now_iso()

        try:
            doc = app.storage.update("profile", _apply)
        except StorageError as exc:
            app.show_dialog("Save failed", str(exc))
            return
        self._render(doc)
        self.ids.toast.flash("Measurements saved")

    def delete_photo(self) -> None:
        app = self._app()
        rel = app.storage.load("profile").get("source_photo")
        for path in ([app.storage.path(rel)] if rel else []) + ([self._session_photo] if self._session_photo else []):
            try:
                os.remove(path)
            except OSError:
                pass
        self._session_photo = ""
        doc = app.storage.update("profile", lambda d: d.__setitem__("source_photo", None))
        self._render(doc)
        self.ids.toast.flash("Photo deleted · measurements kept", icon="delete-outline")

    # -- storage diagnostics ------------------------------------------------
    def refresh_storage_status(self) -> None:
        app = MDApp.get_running_app()
        if app is None or app.storage is None:
            return
        self.ids.storage_path.text = f"Data folder: {app.storage.root}"
        lines = []
        for rel, exists, size in app.storage.status():
            badge = "[color=16A34A][b]OK[/b][/color]" if exists else "[color=DC2626][b]MISSING[/b][/color]"
            size_txt = f"{size:,} B" if exists else "-"
            lines.append(f"{badge}  {rel}  ({size_txt})")
        self.ids.file_status.text = "\n".join(lines)

    def run_self_test(self) -> None:
        app = MDApp.get_running_app()
        ok, message = app.storage.self_test()
        app.show_dialog("Self-test passed" if ok else "Self-test FAILED", message)
        self.refresh_storage_status()


class WelcomeScreen(MDScreen):
    """First-launch onboarding screen."""


class ShellScreen(MDScreen):
    """Main shell: top app bar + bottom navigation with the 4 core zones."""


ZONE_TITLES: Dict[str, str] = {
    "scanner": "Scanner",
    "closet": "Closet",
    "stylist": "AI Stylist",
    "profile": "Profile",
}


def _token(hex_code: str) -> List[float]:
    return list(get_color_from_hex(hex_code))


class WardrobeApp(MDApp):
    """Application root.

    ``self.storage`` is the shared StorageManager, ``self.vision`` the shared
    VisionEngine (None if OpenCV is missing). ``c_*`` are the live colour
    tokens every widget binds to.
    """

    storage = ObjectProperty(None, allownone=True)
    vision = ObjectProperty(None, allownone=True)
    profiler = ObjectProperty(None, allownone=True)
    camera_target = StringProperty("scanner")

    c_bg = ColorProperty(_token(PALETTES["Light"]["bg"]))
    c_surface = ColorProperty(_token(PALETTES["Light"]["surface"]))
    c_surface_alt = ColorProperty(_token(PALETTES["Light"]["surface_alt"]))
    c_frame = ColorProperty(_token(PALETTES["Light"]["frame"]))
    c_text = ColorProperty(_token(PALETTES["Light"]["text"]))
    c_muted = ColorProperty(_token(PALETTES["Light"]["muted"]))
    c_accent = ColorProperty(_token(PALETTES["Light"]["accent"]))
    c_accent_soft = ColorProperty(_token(PALETTES["Light"]["accent_soft"]))
    c_border = ColorProperty(_token(PALETTES["Light"]["border"]))
    c_on_accent = ColorProperty(_token(PALETTES["Light"]["on_accent"]))
    c_overlay = ColorProperty(_token(PALETTES["Light"]["overlay"]))
    c_laser = ColorProperty(_token(PALETTES["Light"]["laser"]))
    c_spot = ColorProperty(_token(PALETTES["Light"]["spot"]))
    current_zone = StringProperty("scanner")
    theme_icon = StringProperty("weather-night")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.title = "Wardrobe AI"
        self._dialog: Optional[MDDialog] = None
        self._theme_anim: Optional[Animation] = None
        self._file_manager: Optional[MDFileManager] = None
        self._android_picker: Optional[AndroidImagePicker] = None
        self._pick_callback: Optional[Callable[[str, bool, int], None]] = None

    # -- build ------------------------------------------------------------
    def build(self) -> MDScreenManager:
        # user_data_dir is only valid once the App object exists.
        self.storage = StorageManager(self.user_data_dir)
        self.storage.initialize()
        try:
            self.storage.purge_orphan_images()
        except (OSError, StorageError):
            Logger.exception(f"{LOG_TAG}: orphan purge failed")

        if CV_AVAILABLE:
            self.vision = VisionEngine(MODELS_DIR)
            self.profiler = BodyProfiler(self.vision, MODELS_DIR)
        else:
            Logger.warning(f"{LOG_TAG}: vision engine disabled: {CV_IMPORT_ERROR}")

        self.bind(c_bg=self._sync_window_color)
        style = self.storage.get_setting("theme.style", "Light")
        self.apply_theme(style if style in PALETTES else "Light", animate=False)
        Builder.load_string(KV)

        manager = MDScreenManager(transition=FadeTransition(duration=0.2))
        manager.add_widget(WelcomeScreen())
        manager.add_widget(ShellScreen())
        manager.add_widget(CameraScreen())

        onboarded = bool(self.storage.get_setting("app.first_run_complete", False))
        manager.current = "shell" if onboarded else "welcome"

        self.storage.set_setting("app.last_launch", utc_now_iso())
        Window.bind(on_keyboard=self._on_keyboard)
        return manager

    # -- theming ----------------------------------------------------------
    def _sync_window_color(self, _inst: Any, value: List[float]) -> None:
        Window.clearcolor = value

    def apply_theme(self, style: str, animate: bool = True) -> None:
        """Switch every colour token to ``style`` (Light/Dark).

        Animated switches use a snapshot cross-fade: the current screen is
        rendered to a texture and laid over the window, the tokens switch in a
        single pass underneath, and the snapshot fades out over 350 ms.
        Interpolating the tokens themselves would re-evaluate every colour
        binding in the app on every frame (~0.5 s per frame on slow devices).
        """
        snapshot = self._theme_snapshot() if animate else None

        self.theme_cls.material_style = "M3"
        self.theme_cls.theme_style = style
        self.theme_cls.primary_palette = MD_PRIMARY[style]
        self.theme_cls.accent_palette = "Amber"
        self.theme_icon = "white-balance-sunny" if style == "Dark" else "weather-night"
        for name, hex_code in PALETTES[style].items():
            setattr(self, f"c_{name}", _token(hex_code))

        if snapshot is not None:
            Window.add_widget(snapshot)
            Clock.schedule_once(lambda _dt: self._crossfade(snapshot), 0)

    def _crossfade(self, old_snap: Widget) -> None:
        """Fade old -> new using two textures only. The live widget tree is
        detached from the window during the 350 ms fade, so each frame draws
        two full-screen quads instead of re-rendering every widget."""
        new_snap = self._theme_snapshot()
        root = self.root
        win_canvas = Window.canvas
        index = win_canvas.indexof(root.canvas) if root is not None else -1
        if new_snap is not None and index > -1:
            Window.remove_widget(old_snap)
            Window.add_widget(new_snap)
            Window.add_widget(old_snap)
            win_canvas.remove(root.canvas)

        def finish(*_args: Any) -> None:
            if new_snap is not None and index > -1 and win_canvas.indexof(root.canvas) == -1:
                win_canvas.insert(index, root.canvas)
            for snap in (old_snap, new_snap):
                if snap is not None and snap.parent is not None:
                    Window.remove_widget(snap)

        anim = Animation(opacity=0.0, d=0.35, t="in_out_quad")
        anim.bind(on_complete=finish)
        anim.start(old_snap)

    def _theme_snapshot(self) -> Optional[Widget]:
        """Render the current screen into a texture-backed widget."""
        from kivy.graphics import ClearBuffers, ClearColor, Fbo, Translate
        from kivy.uix.image import Image
        root = self.root
        if root is None or root.width < 2 or root.height < 2:
            return None
        try:
            parent_canvas = root.parent.canvas if root.parent is not None else None
            index = parent_canvas.indexof(root.canvas) if parent_canvas is not None else -1
            if index > -1:
                parent_canvas.remove(root.canvas)
            fbo = Fbo(size=(int(root.width), int(root.height)), with_stencilbuffer=True)
            with fbo:
                ClearColor(*self.c_bg)
                ClearBuffers()
                Translate(-root.x, -root.y, 0)
            fbo.add(root.canvas)
            fbo.draw()
            texture = fbo.texture
            fbo.remove(root.canvas)
            if index > -1:
                parent_canvas.insert(index, root.canvas)
        except Exception:  # noqa: BLE001 - a missing fade must never break the switch
            Logger.exception(f"{LOG_TAG}: theme snapshot failed")
            return None
        snap = Image(texture=texture, fit_mode="fill", size_hint=(None, None),
                     size=root.size, pos=root.pos)
        snap.on_touch_down = lambda *_a: True  # swallow taps during the fade
        return snap

    def toggle_theme(self) -> None:
        new_style = "Dark" if self.theme_cls.theme_style == "Light" else "Light"
        self.apply_theme(new_style, animate=True)
        self.storage.set_setting("theme.style", new_style)

    # -- navigation / actions ----------------------------------------------
    def scanner_zone(self) -> ScannerZone:
        return self.root.get_screen("shell").ids.scanner_zone

    def profile_zone(self) -> ProfileZone:
        return self.root.get_screen("shell").ids.profile_zone

    # -- shared capture helpers (Scanner + Profile) --------------------------
    def open_camera_for(self, target: str) -> None:
        """Open the in-app camera; the capture is routed to ``target``."""
        self.camera_target = target
        self.root.current = "camera"

    def on_camera_capture(self, path: str) -> None:
        if self.camera_target == "profile":
            self.profile_zone().start_analysis(path, cleanup=True)
        else:
            self.scanner_zone().start_processing(path, cleanup=True)

    def pick_image(self, on_picked: Callable[[str, bool, int], None]) -> None:
        """Let the user choose a photo. ``on_picked(path, cleanup, exif_orientation)``
        - cleanup is True when ``path`` is a private temporary copy."""
        if IS_ANDROID:
            self._android_picker = AndroidImagePicker(
                dest_dir=self.storage.path("cache"),
                on_picked=lambda path, orient: on_picked(path, True, orient),
                on_cancel=lambda: None,
                on_error=lambda msg: self.show_dialog("Upload photo", msg),
            )
            try:
                self._android_picker.open()
            except Exception as exc:  # noqa: BLE001 - Java bridge errors
                Logger.exception(f"{LOG_TAG}: could not open Android picker")
                self.show_dialog("Upload photo", f"Could not open the image picker: {exc}")
            return

        self._pick_callback = on_picked
        if self._file_manager is None:
            self._file_manager = MDFileManager(
                exit_manager=self._close_file_manager,
                select_path=self._on_file_selected,
                ext=list(IMAGE_EXTS) + [e.upper() for e in IMAGE_EXTS],
                preview=False,
            )
        start = os.path.expanduser("~")
        pictures = os.path.join(start, "Pictures")
        self._file_manager.show(pictures if os.path.isdir(pictures) else start)

    def _close_file_manager(self, *_args: Any) -> None:
        if self._file_manager is not None:
            self._file_manager.close()

    def _on_file_selected(self, path: str) -> None:
        self._close_file_manager()
        callback, self._pick_callback = self._pick_callback, None
        if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in IMAGE_EXTS:
            self.show_dialog("Upload photo", "Please choose a JPG, PNG, WebP or BMP image.")
            return
        if callback is not None:
            callback(path, False, 1)

    def complete_onboarding(self) -> None:
        self.storage.set_setting("app.first_run_complete", True)
        self.root.current = "shell"

    def switch_zone(self, zone_name: str) -> None:
        """Floating-dock navigation between the four main zones."""
        if zone_name not in ZONE_TITLES:
            return
        shell = self.root.get_screen("shell")
        if shell.ids.zones.current != zone_name:
            shell.ids.zones.current = zone_name
        self.current_zone = zone_name
        self.on_zone_selected(zone_name)

    def on_zone_selected(self, zone_name: str) -> None:
        shell = self.root.get_screen("shell")
        shell.ids.top_bar.title = ZONE_TITLES.get(zone_name, "Wardrobe AI")
        if zone_name == "profile":
            shell.ids.profile_zone.refresh()
        elif zone_name == "stylist":
            shell.ids.stylist_zone.refresh()
        elif zone_name == "scanner":
            shell.ids.scanner_zone.refresh_engine_status()

    def show_dialog(self, title: str, text: str) -> None:
        if self._dialog is not None:
            self._dialog.dismiss()
        self._dialog = MDDialog(
            title=title,
            text=text,
            buttons=[MDFlatButton(text="OK", on_release=lambda *_args: self._dialog.dismiss())],
        )
        self._dialog.open()

    def _on_keyboard(self, _window: Any, key: int, *_args: Any) -> bool:
        # 27 = Escape on desktop / Back on Android.
        if key == 27 and self.root is not None and self.root.current == "camera":
            self.root.get_screen("camera").close()
            return True
        return False

    # -- Android lifecycle -------------------------------------------------
    def on_pause(self) -> bool:
        # Returning True lets Android suspend the app instead of killing it.
        return True

    def on_resume(self) -> None:
        pass


if __name__ == "__main__":
    WardrobeApp().run()
