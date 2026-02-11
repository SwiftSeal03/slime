import atexit
import logging
import os
import threading
import time
from typing import Any, TextIO

from .misc import SingletonMeta

__all__ = ["TimestampLogger", "timestamp"]

logger = logging.getLogger(__name__)


def _timestamp_file_path(args: Any) -> str | None:
    """Path to this process's timestamp file: <timestamp_path>/<name>.txt."""
    if not args:
        return None
    path = getattr(args, "timestamp_path", None)
    if not path:
        return None
    process = getattr(args, "timestamp_process", "main")
    if process == "actor":
        rank = getattr(args, "timestamp_actor_rank", 0)
        name = f"actor-{rank}"
    else:
        name = process  # main, rollout
    return os.path.join(path, f"{name}.txt")


class TimestampLogger(metaclass=SingletonMeta):
    """
    Singleton that records unix timestamps. Thread-safe within a single process.
    - immediate: keeps a file pointer open, writes and flushes each log line.
    - flush_on_exit: buffers lines in memory, registers atexit to write and flush on program exit.

    All timestamp files are written under the directory given by --timestamp-path, with one file
    per process: main.txt (driver), actor-0.txt, actor-1.txt, ... (training ranks), rollout.txt.
    No cross-process races.
    """

    def __init__(self, args: Any = None):
        self._args = args
        self._file: TextIO | None = None  # used by immediate mode
        self._buffer: list[str] = []  # used by flush_on_exit mode
        self._atexit_registered = False
        self._lock = threading.Lock()

    def _ensure_args(self, args: Any) -> None:
        if self._args is None and args is not None:
            self._args = args

    def _get_path(self) -> str | None:
        return _timestamp_file_path(self._args)

    def _ensure_dir(self, path: str) -> None:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

    def _flush_on_exit(self) -> None:
        with self._lock:
            path = self._get_path()
            buffer_snapshot = list(self._buffer)
        if not path or not buffer_snapshot:
            return
        try:
            self._ensure_dir(path)
            with open(path, "w") as f:  # erase if already exists
                f.writelines(buffer_snapshot)
                f.flush()
        except Exception:  # noqa: S110
            logger.exception("Failed to flush timestamp buffer to %s", path)

    def log(self, msg: str, args: Any = None) -> None:
        with self._lock:
            if args is not None:
                self._ensure_args(args)
            if self._args is None:
                return
            mode = getattr(self._args, "timestamp_mode", "off")
            if mode == "off":
                return
            ts = time.time()
            line = f"{ts}\t{msg}\n"
            if mode == "print":
                logger.info("timestamp %s %s", ts, msg)
                return
            path = self._get_path()
            if not path:
                return
            if mode == "immediate":
                if self._file is None:
                    self._ensure_dir(path)
                    self._file = open(path, "w")  # erase if already exists
                    atexit.register(self._close_file)
                self._file.write(line)
                self._file.flush()
            elif mode == "flush_on_exit":
                self._buffer.append(line)
                if not self._atexit_registered:
                    atexit.register(self._flush_on_exit)
                    self._atexit_registered = True

    def _close_file(self) -> None:
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:  # noqa: S110
                    logger.exception("Failed to close timestamp file")
                self._file = None


def timestamp(args: Any, msg: str) -> None:
    """
    Log a unix timestamp with an optional message. Uses a singleton initialized with args.
    Behavior is controlled by args.timestamp_mode:
    - off: no output
    - print: log via logger only
    - immediate: keep file open, write and flush each line immediately under args.timestamp_path
    - flush_on_exit: buffer lines, write to <timestamp_path>/<name>.txt on program exit
    """
    TimestampLogger(args).log(msg, args=args)
