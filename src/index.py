# src/index.py
"""이미 만든 브리핑 날짜의 자막을 채팅 검색용으로 색인한다 (브리핑은 다시 만들지 않음).

사용법: uv run python -m src.index 2026-09-30 [2026-09-29 ...]   (날짜를 빼면 저장된 모든 날짜)
        --force 를 붙이면 이미 색인된 영상도 다시 색인
"""
import sys

from dotenv import load_dotenv

load_dotenv(".env")

from src.nodes import _log, index_day  # noqa: E402
from src.storage import list_days, load_day  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    for day in args or list_days():
        data = load_day(day)
        if data is None:
            _log(f"{day}: 브리핑 데이터 없음, 건너뜀")
            continue
        _log(f"{day}: 영상 {len(data['summaries'])}개 색인")
        n = index_day(day, data["summaries"], force=force)
        _log(f"{day}: 새로 저장한 청크 {n}개")


if __name__ == "__main__":
    main()
