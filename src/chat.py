# src/chat.py
"""저장된 브리핑과 자막 청크를 근거로 그날 시장에 대해 대화하는 채팅 그래프.

classify → (세부 질문이면) retrieve → answer
- 전체 요약 질문: 그날 브리핑 JSON(오늘의 숫자 + 브리핑 + 영상 목록)을 근거로 답한다.
- 세부 질문: ChromaDB에서 검색한 자막 청크를 근거로 답하고, youtu.be 타임스탬프 링크를 붙인다.
- 여러 날짜 질문: 날짜마다 근거를 나눠 넣고 날짜를 밝혀 비교한다.
- 대화일지 저장 요청: classify → ask_scope(범위 질문) → 사용자 답 → classify → save_log
  (시장일마다 data/chats/날짜_대화요약.md 하나, 다시 저장하면 새 질문만 이어 붙임)
"""
import operator
import re
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from src.config import CHAT_MODEL, KST, SELECT_MODEL
from src.facts import render_numbers
from src.storage import list_days, load_day
from src.vectorstore import mmss, search

HISTORY_MESSAGES = 10  # 답변 생성에 넘길 최근 대화 수
RETRIEVE_K = 8  # 검색할 청크 총수 (날짜가 여러 개면 날짜마다 나눠 찾되 최소 MIN_K_PER_DATE개)
MIN_K_PER_DATE = 4
MAX_DATES = 5  # 한 질문에서 다룰 수 있는 최대 날짜 수


class ChatState(MessagesState):
    dates: list[str]  # 이 질문이 가리키는 시장일 (비교 질문이면 여러 개)
    mode: str
    query: str
    chunks: list[dict]
    turns: Annotated[list[dict], operator.add]  # 이 대화창의 질문·답변 기록 (대화일지 저장용)
    pending_save: dict | None  # 저장 범위를 물어본 뒤 답을 기다리는 중이면 {"dates": [...]}
    save_scope: str  # "thread"(이 대화창) / "all"(그 시장일의 모든 대화창)


# ---------- classify ----------

class _Route(BaseModel):
    mode: Literal["overview", "detail", "save"] = Field(description=(
        "overview: 그날 시장 전체 요약·핵심 이슈·분위기를 묻는 질문 (여러 날짜의 전체 흐름 비교 포함). "
        "detail: 특정 종목·업종·인물 발언·이유·전망·수치의 근거 등 세부 내용을 묻는 질문. "
        "save: 지금까지 나눈 대화를 대화일지·대화요약으로 저장해 달라는 요청"
    ))
    dates: list[str] = Field(description=(
        "질문 문장에 직접 적힌 시장일만 YYYY-MM-DD로 (예: '10월 1일', '9/30', '어제'). "
        "'그날', '그때'처럼 앞 대화를 가리키는 말이나 날짜가 없는 질문이면 빈 목록. 직전 대화의 날짜는 넣지 않는다"
    ))
    search_query: str = Field(description="자막 검색에 쓸 한국어 검색 문장. 대명사를 풀어 쓰고 핵심 종목·주제를 포함 (날짜는 빼고)")


def _youtu(video_id: str) -> str:
    return f"https://youtu.be/{video_id}"


def _previous_dates(state: ChatState) -> list[str]:
    """직전 질문의 날짜. 예전 형식(date 하나)으로 저장된 대화도 읽는다."""
    return state.get("dates") or ([state["date"]] if state.get("date") else [])


# 직전 날짜와 견주는 질문인지 (이때만 직전 날짜를 함께 다룬다)
_COMPARE = re.compile(r"비교|차이|대비|보다|견주|달라|다른 점|같은 점|비슷")


def _parse_scope(text: str) -> str | None:
    """저장 범위 질문에 대한 답: 'thread' / 'all' / 'cancel', 알 수 없으면 None."""
    t = text.strip()
    if any(w in t for w in ("취소", "그만", "안 해", "안해", "하지 마")):
        return "cancel"
    if t.startswith("1") or any(w in t for w in ("이 대화", "현재", "여기", "지금 대화", "이것만")):
        return "thread"
    if t.startswith("2") or any(w in t for w in ("모든", "전체", "다른 대화", "전부")):
        return "all"
    return None


def classify(state: ChatState) -> dict:
    # 저장 범위를 물어본 뒤라면 사용자의 답을 해석한다
    if state.get("pending_save"):
        scope = _parse_scope(state["messages"][-1].content)
        if scope == "cancel":
            return {"mode": "save_cancel"}
        if scope:
            return {"mode": "save", "save_scope": scope}
        return {"mode": "save_ask"}  # 알아듣지 못하면 다시 묻는다

    days = list_days()
    previous = _previous_dates(state)
    history = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-6:])
    llm = ChatOpenAI(model=SELECT_MODEL).with_structured_output(_Route)
    route: _Route = llm.invoke(
        f"오늘 날짜(KST): {datetime.now(KST).date().isoformat()}\n"
        f"브리핑이 있는 날짜: {', '.join(days) or '없음'}\n"
        f"직전 대화의 날짜: {', '.join(previous) or '없음'}\n\n"
        f"대화:\n{history}\n\n마지막 사용자 질문을 분류하세요."
    )
    if route.mode == "save":
        # 날짜를 말했으면 그 날짜만, 아니면 이 대화창에서 다룬 날짜 전체를 저장 대상으로
        return {"mode": "save_ask", "pending_save": {"dates": sorted(set(route.dates))}}
    question = state["messages"][-1].content
    if len(set(route.dates)) == 1 and _COMPARE.search(question):
        dates = set(route.dates) | set(previous)  # '10월 1일이랑 비교하면?' → 직전 날짜와 함께
    else:
        dates = set(route.dates or previous or days[:1])  # 날짜가 없으면 직전 날짜를 이어받는다
    dates = sorted(dates)[:MAX_DATES]
    return {"mode": route.mode, "dates": dates, "query": route.search_query, "chunks": []}


def _after_classify(state: ChatState) -> str:
    mode = state["mode"]
    if mode in ("save_ask", "save_cancel"):
        return "ask_scope"
    if mode == "save":
        return "save_log"
    return "retrieve" if mode == "detail" and state.get("dates") else "answer"


# ---------- retrieve ----------

def retrieve(state: ChatState) -> dict:
    """날짜마다 따로 검색한다 (한 날짜의 청크가 다른 날짜를 밀어내지 않도록)."""
    dates = state["dates"]
    k = max(MIN_K_PER_DATE, RETRIEVE_K // len(dates))
    return {"chunks": [c for d in dates for c in search(state["query"], date=d, k=k)]}


# ---------- answer ----------

_RULES = """당신은 한국 금융·경제 유튜브 방송을 정리해 주는 시장 상담 어시스턴트입니다.

규칙:
- 아래 '근거'에 있는 내용만으로 답합니다. 근거에 없으면 "그날 방송에서는 다루지 않았다"고 말하고 추측하지 않습니다.
- '오늘의 숫자'(KRX·ECOS 확정치)에 있는 지수·수급·환율·종목 등락률은 방송 속 숫자보다 우선합니다.
- 사실(발표된 지표, 실적, 정책)과 출연자 견해를 구분하고, 견해는 누구의 것인지 밝힙니다.
- 자막 원문을 길게 옮겨 적지 말고 핵심을 정리해 전달합니다. 인용은 짧은 구절만 씁니다.
- 근거가 여러 날짜면 날짜별로 구분해 쓰고, 비교를 물으면 공통점과 차이를 정리합니다. 다른 날짜의 숫자를 섞지 않습니다.
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
    dates = state.get("dates") or []
    loaded = {d: load_day(d) for d in dates}
    missing = [d for d, data in loaded.items() if data is None]
    available = {d: data for d, data in loaded.items() if data is not None}
    if not available:
        days = ", ".join(sorted(list_days())) or "없음"
        asked = ", ".join(dates) or "해당 날짜"
        return {"messages": [AIMessage(content=f"{asked} 브리핑이 없습니다. 대화할 수 있는 날짜: {days}")]}

    detail = state["mode"] == "detail"
    sections = []
    for d, data in available.items():
        body = _detail_context(data, [c for c in state.get("chunks") or [] if c["date"] == d]) if detail else _overview_context(data)
        sections.append(f"## [{d}]\n{body}")
    if missing:
        sections.append(f"(브리핑이 없는 날짜: {', '.join(missing)} — 이 날짜는 근거가 없다고 밝힐 것)")
    kind = "세부 질문 — 검색된 자막 청크를 근거로 답하세요." if detail else "전체 요약 질문 — 브리핑을 근거로 답하세요."
    system = SystemMessage(content=(
        f"{_RULES}\n\n기준 날짜: {', '.join(dates)}\n질문 유형: {kind}\n\n# 근거\n" + "\n\n".join(sections)
    ))
    reply = ChatOpenAI(model=CHAT_MODEL).invoke([system, *state["messages"][-HISTORY_MESSAGES:]])
    turn = {
        "asked_at": datetime.now(KST).isoformat(),
        "question": state["messages"][-1].content,
        "answer": reply.content,
        "dates": dates,
        "mode": state["mode"],
    }
    return {"messages": [AIMessage(content=reply.content)], "turns": [turn]}


# ---------- 대화일지 저장 ----------

def ask_scope(state: ChatState) -> dict:
    """저장 범위를 묻는다 (저장 요청마다). 취소면 저장 대기를 푼다."""
    if state["mode"] == "save_cancel":
        return {"messages": [AIMessage(content="대화일지 저장을 취소했어요.")], "pending_save": None}
    dates = (state.get("pending_save") or {}).get("dates") or sorted({d for t in state.get("turns") or [] for d in t["dates"]})
    target = ", ".join(dates) if dates else "이 대화에서 다룬 날짜"
    question = (
        f"대화일지를 어떤 범위로 저장할까요? (대상 시장일: {target})\n\n"
        "1. 이 대화창의 대화만\n"
        "2. 해당 시장일에 나눈 모든 대화 (다른 대화창 포함)\n\n"
        "번호로 답하거나, 저장하지 않으려면 '취소'라고 해주세요."
    )
    return {"messages": [AIMessage(content=question)], "pending_save": state.get("pending_save") or {"dates": dates}}


def _thread_conversations(turns: list[dict], market_date: str) -> list[dict]:
    """이 대화창 기록 중 market_date를 다룬 질문 (같은 질문을 반복했으면 마지막 것만)."""
    unique: dict[str, dict] = {}
    for t in turns:
        if market_date in t["dates"]:
            unique[re.sub(r"\s+", "", t["question"])] = {
                "asked_at": datetime.fromisoformat(t["asked_at"]),
                "question": t["question"].strip(),
                "answer": t["answer"],
                "mode": t["mode"],
                "other_dates": [d for d in t["dates"] if d != market_date],
            }
    return sorted(unique.values(), key=lambda c: c["asked_at"])


def save_log(state: ChatState) -> dict:
    """대화를 시장일별 파일(data/chats/날짜_대화요약.md)에 이어서 저장한다. 이미 저장한 질문은 건너뛴다."""
    from src.chat_summary import append_summary, fetch_conversations

    scope = state.get("save_scope") or "thread"
    turns = state.get("turns") or []
    dates = (state.get("pending_save") or {}).get("dates") or sorted({d for t in turns for d in t["dates"]})
    briefed = set(list_days())
    label = "이 대화창" if scope == "thread" else "모든 대화창"

    lines = []
    for d in dates:
        if d not in briefed:
            lines.append(f"- {d}: 브리핑이 없는 날짜라 저장하지 않았어요.")
            continue
        conversations = _thread_conversations(turns, d) if scope == "thread" else fetch_conversations(d)
        if not conversations:
            lines.append(f"- {d}: 저장할 대화가 없어요.")
            continue
        path, added = append_summary(d, conversations, f"{label} (채팅에서 저장 요청)")
        lines.append(f"- {d}: 새 질문 {added}개 추가 → `{path.as_posix()}`" if added
                     else f"- {d}: 이미 모두 저장된 대화예요 → `{path.as_posix()}`")
    if not lines:
        lines.append("- 저장할 대화가 없어요. 먼저 시장에 대해 질문해 주세요.")
    reply = f"대화일지를 저장했어요 ({label}).\n\n" + "\n".join(lines)
    return {"messages": [AIMessage(content=reply)], "pending_save": None}


# ---------- 그래프 ----------

builder = StateGraph(ChatState)
builder.add_node("classify", classify)
builder.add_node("retrieve", retrieve)
builder.add_node("answer", answer)
builder.add_node("ask_scope", ask_scope)
builder.add_node("save_log", save_log)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", _after_classify, ["retrieve", "answer", "ask_scope", "save_log"])
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
builder.add_edge("ask_scope", END)
builder.add_edge("save_log", END)

graph = builder.compile()
