"""Checksummed WAL-before-data undo log for buffered transaction writes."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclasses.dataclass(frozen=True)
class UndoRecord:
    lsn: int
    txn_id: str
    kind: str
    payload: Dict[str, Any]
    checksum: str


class UndoLog:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._mutex = threading.RLock()
        self._records: List[UndoRecord] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists():
                self._load()

    def append(self, txn_id: str, kind: str, payload: Dict[str, Any] | None = None) -> UndoRecord:
        with self._mutex:
            row = {
                "lsn": len(self._records) + 1,
                "txn_id": str(txn_id),
                "kind": str(kind).upper(),
                "payload": dict(payload or {}),
            }
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            record = UndoRecord(**row, checksum=hashlib.sha256(encoded).hexdigest())
            self._records.append(record)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            return record

    def begin(self, txn_id: str) -> UndoRecord:
        return self.append(txn_id, "BEGIN")

    def update(self, txn_id: str, *, object_id: str, old_value: str, old_version: int) -> UndoRecord:
        return self.append(
            txn_id,
            "UPDATE",
            {"object_id": str(object_id), "old_value": str(old_value), "old_version": int(old_version)},
        )

    def update_batch(
        self,
        txn_id: str,
        updates: Iterable[tuple[str, str, int]],
    ) -> UndoRecord | None:
        rows = [
            {
                "object_id": str(object_id),
                "old_value": str(old_value),
                "old_version": int(old_version),
            }
            for object_id, old_value, old_version in updates
        ]
        if not rows:
            return None
        return self.append(txn_id, "UPDATE_BATCH", {"updates": rows})

    def commit(self, txn_id: str) -> UndoRecord:
        return self.append(txn_id, "COMMIT")

    def abort(self, txn_id: str) -> UndoRecord:
        return self.append(txn_id, "ABORT")

    def incomplete(self) -> Dict[str, List[UndoRecord]]:
        grouped: Dict[str, List[UndoRecord]] = {}
        finished = set()
        for record in self._records:
            grouped.setdefault(record.txn_id, []).append(record)
            if record.kind in {"COMMIT", "ABORT"}:
                finished.add(record.txn_id)
        return {txn_id: records for txn_id, records in grouped.items() if txn_id not in finished}

    def recover(self, store: Any) -> list[str]:
        recovered = []
        for txn_id, records in self.incomplete().items():
            updates = []
            for record in records:
                if record.kind == "UPDATE":
                    updates.append(record.payload)
                elif record.kind == "UPDATE_BATCH":
                    updates.extend(record.payload.get("updates", ()))
            checks = []
            writes = []
            for payload in reversed(updates):
                object_id = str(payload["object_id"])
                old_version = int(payload["old_version"])
                current = store.get(object_id)
                if int(current.version) == old_version + 1:
                    checks.append((object_id, int(current.version)))
                    writes.append((object_id, str(payload["old_value"])))
                elif int(current.version) != old_version:
                    raise RuntimeError(f"cannot safely recover {txn_id}: unexpected version for {object_id}")
            if writes and not store.batch_put_if_version(checks, writes):
                raise RuntimeError(f"atomic undo failed for {txn_id}")
            self.abort(txn_id)
            recovered.append(txn_id)
        return recovered

    def records(self) -> tuple[UndoRecord, ...]:
        return tuple(self._records)

    def _load(self) -> None:
        expected_lsn = 1
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                checksum = str(row.pop("checksum"))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RuntimeError(f"invalid undo log record at line {line_number}") from exc
            if int(row.get("lsn", -1)) != expected_lsn:
                raise RuntimeError(f"undo log LSN discontinuity at line {line_number}")
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if hashlib.sha256(encoded).hexdigest() != checksum:
                raise RuntimeError(f"undo log checksum mismatch at LSN {row.get('lsn')}")
            self._records.append(UndoRecord(**row, checksum=checksum))
            expected_lsn += 1
