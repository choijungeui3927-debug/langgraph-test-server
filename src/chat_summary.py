# src/chat_summary.py
"""채팅 대화 요약본 저장: 특정 '시장일'을 기준으로 나눈 대화를 모아 요약한다.

대화한 날이 아니라 채팅이 기준으로 삼은 시장일(chat 그래프의 date)로 대화를 모은다.
LangSmith 실행 기록에서 질문과 답변을 가져오고, 종목 등락률은 확정 숫자(KRX)를 코드로 넣는다.

사용법: uv run python -m src.chat_summary 2026-10-06
결과:   data/chats/2026-10-06_대화요약.md (시장일마다 파일 하나. 다시 저장하면 아직 저장 안 한 질문만 "
자동:   uv run python -m src.chat_summary --auto  (마지막 자동 실행 이후 새 대화가 있는 시장일만 요약, chat_summary_auto.bat로 매일 예약)
"""
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

import httpx  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.config import DATA_DIR, KST, SUMMARY_MODEL  # noqa: E402
from src.facts import stock_change, ticker_names  # noqa: E402
from src.storage import list_days, load_day  # noqa: E402

CHAT_DIR = Path(DATA_DIR) / "chats"
LOOKBACK_DAYS = 60  # 시장일 이후 이 기간 안에 나눈 대화를 찾는다


# ---------- LangSmith에서 대화 가져오기 ----------

def _content(message) -> tuple[str, str]:
    """직렬화된 메시지에서 (역할, 내용)을 꺼낸다."""
    if not isinstance(message, dict):
        return "", str(message)
    body = message.get("kwargs", message)
    role = body.get("type") or body.get("role") or ""
    content = body.get("content", "")
    if isinstance(content, list):
        content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return role, content


def fetch_conversations(market_date: str) -> list[dict]:
    """시장일이 market_date인 채팅 실행의 질문·답변 목록 (시간순)."""
    endpoint, key = os.environ["LANGSMITH_ENDPOINT"], os.environ["LANGSMITH_API_KEY"]
    headers = {"x-api-key": key}
    project = os.environ["LANGSMITH_PROJECT"]
    session_id = httpx.get(f"{endpoint}/api/v1/sessions", headers=headers, params={"name": project}, timeout=30).json()[0]["id"]

    start = (date.fromisoformat(market_date) - timedelta(days=1)).isoformat()
    end = (date.fromisoformat(market_date) + timedelta(days=LOOKBACK_DAYS)).isoformat()
    body = {"session": [session_id], "is_root": True, "limit": 100,
            "start_time": f"{start}T00:00:00", "end_time": f"{end}T00:00:00",
            "select": ["start_time", "inputs", "outputs"]}
    runs, cursor = [], None
    while True:
        page = httpx.post(f"{endpoint}/api/v1/runs/query", headers=headers,
                          json={**body, **({"cursor": cursor} if cursor else {})}, timeout=60).json()
        runs += page.get("runs", [])
        cursor = (page.get("cursors") or {}).get("next")
        if not cursor:
            break

    conversations = []
    for run in runs:
        outputs = run.get("outputs") or {}
        # 여러 날짜 질문(dates)은 각 날짜 요약에 모두 넣는다. 예전 기록은 date 하나만 있다
        dates = outputs.get("dates") or ([outputs["date"]] if outputs.get("date") else [])
        if "messages" not in (run.get("inputs") or {}) or market_date not in dates:
            continue
        questions = [c for r, c in map(_content, run["inputs"]["messages"]) if r in ("human", "user")]
        answers = [c for r, c in map(_content, outputs.get("messages") or []) if r in ("ai", "assistant")]
        if questions and answers:
            asked_at = datetime.fromisoformat(run["start_time"].replace("Z", "+00:00")).astimezone(KST)
            conversations.append({"asked_at": asked_at, "question": questions[-1].strip(),
                                  "answer": answers[-1], "mode": outputs.get("mode"),
                                  "other_dates": [d for d in dates if d != market_date]})
    conversations.sort(key=lambda c: c["asked_at"])

    # 같은 질문을 반복한 경우 마지막 답변 하나만 남긴다
    unique: dict[str, dict] = {}
    for c in conversations:
        unique[re.sub(r"\s+", "", c["question"])] = c
    return sorted(unique.values(), key=lambda c: c["asked_at"])


# ---------- LLM 요약 ----------

class _QA(BaseModel):
    number: int = Field(description="대화의 [질문 N] 번호")
    question: str = Field(description="사용자 질문 (원문을 다듬어 한 줄로)")
    points: list[str] = Field(description="답변의 핵심 2~3개. 수치·고유명사 유지, 출연자 견해면 누구의 견해인지 표시")
    sources: list[str] = Field(description="답변에 있던 출처 마크다운 링크를 글자 그대로 복사 (최대 3개). 새로 만들지 않음")


class _ChannelView(BaseModel):
    channel: str
    view: str = Field(description="이 채널이 해당 주제를 어떻게 봤는지 한두 문장")


class _Topic(BaseModel):
    topic: str = Field(description="여러 채널이 함께 다룬 주제 (예: 환율, 외국인 수급, 반도체 소부장)")
    views: list[_ChannelView]


class _Digest(BaseModel):
    qas: list[_QA]
    stocks: list[str] = Field(description="대화 답변에 나온 상장 종목명 (정식 이름)")
    sectors: list[str] = Field(description="대화 답변에 나온 섹터·테마 (예: 우주항공, 2차전지, 반도체 소부장)")
    channel_views: list[_Topic] = Field(description="두 개 이상 채널의 시각이 답변에 함께 나온 주제만. 없으면 빈 목록")


_PROMPT = """다음은 {date} 국내 증시에 대해 사용자가 시장 브리핑 챗봇과 나눈 대화입니다.
대화 요약본을 만드세요.

규칙:
- 질문마다 답변의 핵심을 2~3개로 정리합니다. 답변에 없는 내용은 만들지 않습니다.
- '함께 물어본 날짜'가 있는 질문(여러 날짜 비교)은 {date}에 관한 내용을 중심으로 정리하고,
  다른 날짜와의 비교 결과는 한 줄로만 덧붙입니다. 다른 날짜의 숫자를 {date}의 숫자처럼 쓰지 않습니다.
- 출처 링크는 답변에 있던 마크다운 링크를 글자 그대로 복사합니다.
- 종목은 답변에 나온 상장 종목만 정식 이름으로 적습니다. 숫자(등락률)는 적지 않습니다(코드가 따로 넣음).
- 채널별 시각 차이는 답변에 채널 이름이 함께 나온 주제만 정리합니다.

대화:
{conversation}"""


def summarize(market_date: str, conversations: list[dict]) -> _Digest:
    text = "\n\n".join(
        f"[질문 {i}] ({c['asked_at']:%m/%d %H:%M}) {c['question']}"
        + (f"\n(함께 물어본 날짜: {', '.join(c['other_dates'])})" if c["other_dates"] else "")
        + f"\n[답변 {i}]\n{c['answer']}"
        for i, c in enumerate(conversations, 1)
    )
    llm = ChatOpenAI(model=SUMMARY_MODEL).with_structured_output(_Digest)
    digest: _Digest = llm.invoke(_PROMPT.format(date=market_date, conversation=text))
    # 답변에 실제로 있던 링크만 남긴다
    all_answers = "\n".join(c["answer"] for c in conversations)
    for qa in digest.qas:
        qa.sources = [s for s in qa.sources if (m := re.search(r"\]\((https?://[^)]+)\)", s)) and m.group(1) in all_answers]
    # '채널별 시각 차이'는 두 채널 이상이 함께 나온 주제만
    digest.channel_views = [t for t in digest.channel_views if len({v.channel for v in t.views}) >= 2]
    return digest


# ---------- 종목 등락률 (코드로 확정) ----------

def stock_rows(market_date: str, names: list[str]) -> list[tuple[str, str, str]]:
    """(종목, 종가, 등락률). 그날 확정 숫자에 없으면 KRX에서 조회하고, 상장 종목이 아니면 '-'."""
    day = load_day(market_date) or {}
    known = (day.get("facts") or {}).get("stocks") or {}
    target = date.fromisoformat(market_date)
    missing = [n for n in names if n not in known]
    if missing:
        listed = ticker_names(target.strftime("%Y%m%d"))
        known = {**known, **stock_change([n for n in missing if n in listed], target)}
    rows = []
    for n in names:
        x = known.get(n)
        rows.append((n, f"{x['close']:,}원", f"{x['change_pct']:+.2f}%") if x else (n, "-", "-"))
    return rows


# ---------- 저장 (시장일마다 파일 하나, 다시 저장하면 새 질문만 이어서 추가) ----------

SAVED_INDEX = CHAT_DIR / "saved_index.json"  # 시장일별로 이미 저장한 질문 목록 (중복 저장 방지)


def summary_path(market_date: str) -> Path:
    return CHAT_DIR / f"{market_date}_대화요약.md"


def _key(conv: dict) -> str:
    """질문을 구분하는 키: 질문 시각(분) + 공백을 뺀 질문."""
    return f"{conv['asked_at']:%Y%m%d%H%M}|{re.sub(r'\s+', '', conv['question'])}"


def _load_index() -> dict[str, list[str]]:
    return json.loads(SAVED_INDEX.read_text(encoding="utf-8")) if SAVED_INDEX.exists() else {}


def _render_section(n: int, scope: str, conversations: list[dict], digest: _Digest,
                    rows: list[tuple[str, str, str]]) -> str:
    first, last = conversations[0]["asked_at"], conversations[-1]["asked_at"]
    lines = [
        f"## 저장 {n} — {datetime.now(KST):%Y-%m-%d %H:%M} KST · {scope}",
        "",
        f"- 대화: 질문 {len(conversations)}개 ({first:%Y-%m-%d %H:%M} ~ {last:%Y-%m-%d %H:%M} KST)",
        "",
        "### 질문별 정리",
        "",
    ]
    for i, qa in enumerate(digest.qas, 1):
        conv = conversations[qa.number - 1] if 1 <= qa.number <= len(conversations) else {"other_dates": []}
        compared = f" (비교: {', '.join(conv['other_dates'])})" if conv["other_dates"] else ""
        lines.append(f"#### {i}. {qa.question}{compared}")
        lines += [f"- {p}" for p in qa.points]
        if qa.sources:
            lines.append(f"- 출처: {' · '.join(qa.sources)}")
        lines.append("")

    lines += ["### 대화에 나온 종목·섹터", ""]
    if rows:
        lines += ["| 종목 | 종가 | 등락률 |", "|---|---|---|"]
        lines += [f"| {n} | {c} | {p} |" for n, c, p in rows]
        lines += ["", "*종가·등락률: KRX 확정치 (코드로 조회)*", ""]
    if digest.sectors:
        lines += [f"- 섹터·테마: {', '.join(digest.sectors)}", ""]

    lines += ["### 채널별 시각 차이", ""]
    if digest.channel_views:
        for t in digest.channel_views:
            lines.append(f"#### {t.topic}")
            lines += [f"- **{v.channel}**: {v.view}" for v in t.views]
            lines.append("")
    else:
        lines += ["- 여러 채널의 시각이 함께 나온 주제가 없었습니다.", ""]
    return "\n".join(lines)


def append_summary(market_date: str, conversations: list[dict], scope: str) -> tuple[Path, int]:
    """시장일 파일에 아직 저장하지 않은 질문만 요약해 이어 붙인다. (파일 경로, 새로 추가한 질문 수)."""
    path = summary_path(market_date)
    index = _load_index()
    saved = set(index.get(market_date, []))
    new = [c for c in conversations if _key(c) not in saved]
    if not new:
        return path, 0

    digest = summarize(market_date, new)
    rows = stock_rows(market_date, digest.stocks)
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = f"# {market_date} 시장 대화요약\n\n- 시장일: {market_date}\n"
    n = len(re.findall(r"^## 저장 \d+", text, flags=re.M)) + 1
    path.write_text(f"{text.rstrip()}\n\n{_render_section(n, scope, new, digest, rows)}", encoding="utf-8")

    index[market_date] = sorted(saved | {_key(c) for c in new})
    SAVED_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return path, len(new)


def save_summary(market_date: str, scope: str = "모든 대화창") -> tuple[Path, int] | None:
    """시장일의 대화 전체(LangSmith 기록)에서 새 질문만 파일에 이어 붙인다. 대화가 없으면 None."""
    conversations = fetch_conversations(market_date)
    if not conversations:
        return None
    print(f"{market_date} 시장일 대화 {len(conversations)}개 확인 중…", flush=True)
    return append_summary(market_date, conversations, scope)


# ---------- 자동 실행 (매일 예약) ----------

AUTO_STATE = CHAT_DIR / "auto_state.json"


def market_dates_asked_since(since: datetime) -> list[str]:
    """since 이후에 나눈 대화들이 가리킨 시장일 목록."""
    endpoint, key = os.environ["LANGSMITH_ENDPOINT"], os.environ["LANGSMITH_API_KEY"]
    headers = {"x-api-key": key}
    session_id = httpx.get(f"{endpoint}/api/v1/sessions", headers=headers,
                           params={"name": os.environ["LANGSMITH_PROJECT"]}, timeout=30).json()[0]["id"]
    body = {"session": [session_id], "is_root": True, "limit": 100,
            "start_time": since.astimezone(timezone.utc).isoformat(), "select": ["inputs", "outputs"]}
    dates, cursor = set(), None
    while True:
        page = httpx.post(f"{endpoint}/api/v1/runs/query", headers=headers,
                          json={**body, **({"cursor": cursor} if cursor else {})}, timeout=60).json()
        for run in page.get("runs", []):
            out = run.get("outputs") or {}
            if "messages" in (run.get("inputs") or {}):
                dates |= set(out.get("dates") or ([out["date"]] if out.get("date") else []))
        cursor = (page.get("cursors") or {}).get("next")
        if not cursor:
            break
    return sorted(dates)


def run_auto() -> None:
    """마지막 자동 실행 이후 새로 나눈 대화가 있는 시장일마다 요약 파일을 만든다 (그 시장일 대화 전체 기준)."""
    started = datetime.now(KST)
    state = json.loads(AUTO_STATE.read_text(encoding="utf-8")) if AUTO_STATE.exists() else {}
    since = datetime.fromisoformat(state["last_run"]) if state.get("last_run") else started - timedelta(days=1)
    print(f"[{started:%Y-%m-%d %H:%M}] 자동 대화요약: {since:%Y-%m-%d %H:%M} 이후 대화 확인", flush=True)

    asked = market_dates_asked_since(since)
    briefed = set(list_days())
    dates = [d for d in asked if d in briefed]  # 브리핑이 없는 날짜(예: 장이 끝나기 전 '오늘')는 근거가 없어 요약하지 않는다
    for d in sorted(set(asked) - briefed):
        print(f"  {d}: 브리핑이 없는 날짜라 건너뜀", flush=True)
    if not dates:
        print("  요약할 새 대화 없음", flush=True)
    for d in dates:
        result = save_summary(d, scope="모든 대화창 (매일 자동)")
        if result is None:
            print(f"  {d}: 대화 없음", flush=True)
        else:
            print(f"  {d}: 새 질문 {result[1]}개 추가 → {result[0]}" if result[1] else f"  {d}: 새 질문 없음 (이미 저장됨)", flush=True)

    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    AUTO_STATE.write_text(json.dumps({"last_run": started.isoformat()}), encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("사용법: uv run python -m src.chat_summary YYYY-MM-DD   (자동: --auto)")
    if sys.argv[1] == "--auto":
        run_auto()
        return
    market_date = sys.argv[1]
    result = save_summary(market_date, scope="모든 대화창 (명령 실행)")
    if result is None:
        sys.exit(f"{market_date} 시장일 기준 대화가 없습니다.")
    path, added = result
    print(f"새 질문 {added}개 추가: {path}" if added else f"새 질문 없음 (이미 저장됨): {path}")


if __name__ == "__main__":
    main()
