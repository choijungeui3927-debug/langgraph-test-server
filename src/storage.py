# src/storage.py
"""날짜별 요약 결과를 data/YYYY-MM-DD.json 으로 저장하고 읽는다."""
import json
from pathlib import Path

from src.config import DATA_DIR


def _path(target_date: str) -> Path:
    return Path(DATA_DIR) / f"{target_date}.json"


def save_day(target_date: str, videos: list[dict], summaries: list[dict], digest: str) -> None:
    path = _path(target_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"date": target_date, "videos": videos, "summaries": summaries, "digest": digest}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_day(target_date: str) -> dict | None:
    path = _path(target_date)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_days() -> list[str]:
    """저장된 날짜 목록 (최신순)."""
    return sorted((p.stem for p in Path(DATA_DIR).glob("*.json")), reverse=True)
