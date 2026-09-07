# run.py
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# 프로젝트 루트 경로를 sys.path에 추가 (직접 실행 시 모듈 탐색 오류 방지)
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Windows 터미널 출력 인코딩 설정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# .env 환경 변수 로드
load_dotenv(find_dotenv())

from src.app import graph

def main():
    print("🚀 LangGraph 에이전트 대화형 실행 모드 (종료: 'q' 또는 'exit')")

    while True:
        try:
            user_input = input("\n👤 질문/요청 입력: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n프로그램을 종료합니다.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit"):
            print("프로그램을 종료합니다.")
            break

        initial_state = {
            "messages": [{"role": "user", "content": user_input}]
        }

        print("⏳ 에이전트 실행 중...")
        result = graph.invoke(initial_state)

        print("\n[실행 결과]")
        print(f"🤖 생성된 농담: {result.get('joke')}")
        if result.get("score") is not None:
            print(f"⭐ 점수: {result.get('score')}")
        if result.get("feedback"):
            print(f"💬 평가 피드백: {result.get('feedback')}")

if __name__ == "__main__":
    main()