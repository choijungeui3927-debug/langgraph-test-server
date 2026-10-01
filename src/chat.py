# src/chat.py
"""저장된 브리핑과 자막 청크를 근거로 그날 시장에 대해 대화하는 채팅 그래프.

classify → (세부 질문이면) retrieve → answer
- 전체 요약 질문: 그날 브리핑 JSON(오늘의 숫자 + 브리핑 + 영상 목록)을 근거로 답한다.
- 세부 질문: ChromaDB에서 검색한 자막 청크를 근거로 답하고, youtu.be 타임스탬프 링크를 붙인다.
"""
import re
from datetime import datetime
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from src.config import CHAT_MODEL, KST, SELECT_MODEL
from src.facts import render_numbers
from src.storage import list_days, load_day
from src.vectorstore import mmss, search

HISTORY_MESSAGES = 10  # 답변 생성에 넘길 최근 대화 수
RETRIEVE_K = 8


class ChatState(MessagesState):
    date: str | None
    mode: str
    query: str
    chunks: list[dict]


# ---------- classify ----------

class _Route(BaseModel):
    mode: Literal["overview", "detail"] = Field(description=(
        "overview: 그날 시장 전체 요약·핵심 이슈·분위기를 묻는 질문. "
        "detail: 특정 종목·업종·인물 발언·이유·전망·수치의 근거 등 세부 내용을 묻는 질문"
    ))
    date: str | None = Field(description="질문이 가리키는 날짜 YYYY-MM-DD. 언급이 없으면 직전 대화의 날짜, 그것도 없으면 null")
    search_query: str = Field(description="자막 검색에 쓸 한국어 검색 문장. 대명사를 풀어 쓰고 핵심 종목·주제를 포함")


def _youtu(video_id: str) -> str:
    return f"https://youtu.be/{video_id}"


def classify(state: ChatState) -> dict:
    days = list_days()
    history = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-6:])
    llm = ChatOpenAI(model=SELECT_MODEL).with_structured_output(_Route)
    route: _Route = llm.invoke(
        f"오늘 날짜(KST): {datetime.now(KST).date().isoformat()}\n"
        f"브리핑이 있는 날짜: {', '.join(days) or '없음'}\n"
        f"직전 대화의 날짜: {state.get('date') or '없음'}\n\n"
        f"대화:\n{history}\n\n마지막 사용자 질문을 분류하세요."
    )
    date = route.date or state.get("date") or (days[0] if days else None)
    return {"mode": route.mode, "date": date, "query": route.search_query, "chunks": []}


def _after_classify(state: ChatState) -> str:
    return "retrieve" if state["mode"] == "detail" and state.get("date") else "answer"


# ---------- retrieve ----------

def retrieve(state: ChatState) -> dict:
    return {"chunks": search(state["query"], date=state["date"], k=RETRIEVE_K)}


# ---------- answer ----------

_RULES = """당신은 한국 금융·경제 유튜브 방송을 정리해 주는 시장 상담 어시스턴트입니다.

규칙:
- 아래 '근거'에 있는 내용만으로 답합니다. 근거에 없으면 "그날 방송에서는 다루지 않았다"고 말하고 추측하지 않습니다.
- '오늘의 숫자'(KRX·ECOS 확정치)에 있는 지수·수급·환율·종목 등락률은 방송 속 숫자보다 우선합니다.
- 사실(발표된 지표, 실적, 정책)과 출연자 견해를 구분하고, 견해는 누구의 것인지 밝힙니다.
- 자막 원문을 길게 옮겨 적지 말고 핵심을 정리해 전달합니다. 인용은 짧은 구절만 씁니다.
- 출처는 문장 끝에 마크다운 링크로 답니다. 근거에 적힌 링크를 글자 그대로 복사하고 새로 만들지 않습니다.
  - 자막 청크 근거: 청크의 '인용 링크' — [영상 제목 (분:초)](https://youtu.be/…?t=초)
  - 브리핑 근거: [영상 제목](https://youtu.be/…)
- 투자 권유가 아니라 방송 내용 정리라는 점을 유지하고, 한국어로 간결하게 답합니다."""


TITLE_MAX = 40  # 링크 글자로 쓸 영상 제목 길이


def _short(title: str) -> str:
    return title if len(title) <= TITLE_MAX else title[:TITLE_MAX].rstrip() + "…"


def _overview_context(data: dict) -> str:
    """브리핑 본문의 [채널](youtube.com/watch?v=ID) 링크를 [영상 제목](youtu.be/ID)로 바꿔 넘긴다."""
    titles = {s["video_id"]: _short(s["title"]) for s in data["summaries"]}

    def relink(m: re.Match) -> str:
        vid = m.group(2)
        return f"[{titles.get(vid, m.group(1))}]({_youtu(vid)})"

    digest = re.sub(r"\[([^\]]+)\]\(https://www\.youtube\.com/watch\?v=([\w-]{11})\)", relink, data["digest"])
    videos = "\n".join(f"- [{_short(s['title'])}]({_youtu(s['video_id'])}) — {s['channel']}" for s in data["summaries"])
    return f"{digest}\n\n## 브리핑에 쓰인 영상\n{videos}"


def _detail_context(data: dict, chunks: list[dict]) -> str:
    numbers = render_numbers(data["facts"]) if data.get("facts") else "(확정 숫자 없음)"
    found = "\n\n".join(
        f"[{i}] {c['channel']} | {mmss(c['start'])}~{mmss(c['end'])}\n"
        f"인용 링크: [{_short(c['title'])} ({mmss(c['start'])})]({c['url']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    ) or "(검색 결과 없음)"
    return f"{numbers}\n\n## 검색된 자막 청크\n{found}"


def answer(state: ChatState) -> dict:
    date = state.get("date")
    data = load_day(date) if date else None
    if data is None:
        days = ", ".join(list_days()) or "없음"
        return {"messages": [AIMessage(content=f"{date or '해당 날짜'} 브리핑이 없습니다. 대화할 수 있는 날짜: {days}")]}

    if state["mode"] == "detail":
        context = _detail_context(data, state.get("chunks") or [])
        kind = "세부 질문 — 검색된 자막 청크를 근거로 답하세요."
    else:
        context = _overview_context(data)
        kind = "전체 요약 질문 — 브리핑을 근거로 답하세요."
    system = SystemMessage(content=f"{_RULES}\n\n기준 날짜: {date}\n질문 유형: {kind}\n\n# 근거\n{context}")
    reply = ChatOpenAI(model=CHAT_MODEL).invoke([system, *state["messages"][-HISTORY_MESSAGES:]])
    return {"messages": [AIMessage(content=reply.content)]}


# ---------- 그래프 ----------

builder = StateGraph(ChatState)
builder.add_node("classify", classify)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", _after_classify, ["retrieve", "answer"])
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

graph = builder.compile()
