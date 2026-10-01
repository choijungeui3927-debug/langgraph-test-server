# src/storage.py
"""날짜별 요약 결과를 data/YYYY-MM-DD.json 으로 저장하고 읽는다."""
import json
from datetime import datetime
from pathlib import Path

from src.config import DATA_DIR, KST, LOG_DIR


def _path(target_date: str) -> Path:
    return Path(DATA_DIR) / f"{target_date}.json"


def save_day(
    target_date: str,
    videos: list[dict],
    summaries: list[dict],
    digest: str,
    facts: dict | None = None,
    fact_check: dict | None = None,
) -> None:
    path = _path(target_date)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "date": target_date,
        "facts": facts,  # 코드가 확정한 숫자 (KRX, ECOS)
        "fact_check": fact_check,  # 브리핑 숫자 검증 결과
        "videos": videos,
        "summaries": summaries,
        "digest": digest,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(data)


def write_markdown(data: dict) -> Path:
    """사람이 읽기 위한 브리핑 파일 data/YYYY-MM-DD.md 를 만든다 (브리핑 + 사용한 영상 목록)."""
    sources = "\n".join(
        f"- [{s.get('segment') or '일반'}] [{s['channel']}] [{s['title']}]({s['url']}) ({s['published_at'][:16]} KST)"
        + (f"\n  - 선정 이유: {s['selection_reason']}" if s.get("selection_reason") else "")
        for s in data["summaries"]
    )
    path = _path(data["date"]).with_suffix(".md")
    path.write_text(f"{data['digest']}\n\n---\n\n## 사용한 영상\n\n{sources}\n", encoding="utf-8")
    return path


def log_fact_errors(target_date: str, errors: list[str]) -> None:
    """재생성 후에도 남은 숫자 오류를 logs/fact_check.log 에 남긴다."""
    path = Path(LOG_DIR) / "fact_check.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(KST).isoformat(timespec="seconds")
    lines = [f"[{stamp}] {target_date} 재생성 후에도 숫자 오류 {len(errors)}건"] + [f"  - {e}" for e in errors]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(lines[0])


def load_day(target_date: str) -> dict | None:
    path = _path(target_date)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_days() -> list[str]:
    """저장된 날짜 목록 (최신순)."""
    return sorted((p.stem for p in Path(DATA_DIR).glob("*.json")), reverse=True)
