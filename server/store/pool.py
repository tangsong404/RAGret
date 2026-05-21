from __future__ import annotations

import queue
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class SqliteConnectionPool:
    """Thread-safe SQLite connection pool. Each connection uses WAL + foreign_keys."""

    def __init__(self, db_path: Path, min_size: int = 4, max_size: int = 32) -> None:
        self._path = db_path
        self._max = max_size
        self._lock = threading.Lock()
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue()
        self._created = min_size
        for _ in range(min_size):
            self._pool.put(self._new_conn())

    def _new_conn(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def acquire(self) -> Generator[sqlite3.Connection, None, None]:
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._created < self._max:
                    conn = self._new_conn()
                    self._created += 1
                else:
                    conn = self._pool.get()  # block until one is free
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            self._pool.put(conn)

    def close(self) -> None:
        while True:
            try:
                self._pool.get_nowait().close()
            except queue.Empty:
                break
