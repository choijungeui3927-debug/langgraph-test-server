# src/state.py
from langgraph.graph import MessagesState

# MessagesState 상속. 나머지 항목은 필요에 따라서 본인이 추가 가능
class JokeState(MessagesState):
    joke: str          # 생성된 최신 농담 텍스트
    score: int         # 평가 점수 (1~10)
    feedback: str      # 평가 피드백 및 개선점