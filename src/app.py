# src/app.py
'''
from langgraph.graph import StateGraph, START, END

builder = StateGraph()
builder.add_edge(START, END)

# langgraph.json 에 등록된 이름과 맞아야함!
graph = builder.compile()
'''

# src/app.py
import os
import sys
from dotenv import load_dotenv, find_dotenv

# Windows 터미널 출력 인코딩 설정 (이모지 및 한글 출력 에러 방지)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# .env 파일에 설정된 LangSmith 및 API 환경 변수 로드
load_dotenv(find_dotenv())

from langgraph.graph import StateGraph, START, END
from src.state import JokeState
from src.nodes import joke_node, eval_node

# LangSmith 트레이싱 활성화 상태 및 프로젝트 확인
print(f"[LangSmith] Tracing 활성화됨 | 프로젝트명: {os.getenv('LANGSMITH_PROJECT')}")

# 1. JokeState 스키마를 반영한 그래프 빌더 초기화
builder = StateGraph(JokeState)

# 2. 노드 등록 (농담 생성 -> 농담 평가)
builder.add_node("joke_generator", joke_node)
builder.add_node("joke_evaluator", eval_node)

# 3. 엣지 연결 (START -> 농담 생성 -> 농담 평가 -> END)
builder.add_edge(START, "joke_generator")
builder.add_edge("joke_generator", "joke_evaluator")
builder.add_edge("joke_evaluator", END)

# langgraph.json 에 등록된 이름과 맞아야함!
graph = builder.compile()
