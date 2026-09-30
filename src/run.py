# src/run.py
"""로컬 실행: uv run python -m src.run [YYYY-MM-DD]"""
import sys
from collections import Counter

from dotenv import load_dotenv

load_dotenv(".env")

from src.app import graph  # noqa: E402


def main() -> None:
    inputs = {"target_date": sys.argv[1]} if len(sys.argv) > 1 else {}
    # 자막 요청이 한꺼번에 몰리면 유튜브가 IP를 차단할 수 있어 동시 실행 수를 제한
    result = graph.invoke(inputs, config={"max_concurrency": 4})

    print(f"수집 영상 {len(result.get('videos', []))}개 / "
          f"요약 {len(result.get('summaries', []))}개 / "
          f"제외 {len(result.get('skipped', []))}개")
    reasons = Counter(s["reason"] for s in result.get("skipped", []))
    for reason, count in reasons.items():
        print(f"  - {reason}: {count}개")
    print()
    print(result["digest"])


if __name__ == "__main__":
    main()
