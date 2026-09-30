# src/chat.py
"""저장된 날짜별 뉴스(요약·브리핑·자막)를 바탕으로 대화하는 채팅 그래프."""
from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_openai import ChatOpenAI

from src.config import CHAT_MODEL, KST
from src.storage import list_days, load_day
from src.youtube import get_cached_transcript

SNIPPET_RADIUS = 250  # 검색 결과로 보여줄 키워드 앞뒤 글자 수
MAX_SNIPPETS_PER_VIDEO = 3
READ_CHUNK_CHARS = 6000


# ---------- 도구 ----------

def list_available_days() -> str:
    """요약 데이터가 저장된 날짜 목록을 최신순으로 반환한다."""
    days = list_days()
    if not days:
        return "저장된 날짜가 없습니다. 먼저 요약 그래프(agent)를 실행해야 합니다."
    return "\n".join(f"- {d} (영상 {len(load_day(d)['summaries'])}개)" for d in days)


def get_day_overview(date: str) -> str:
    """해당 날짜(YYYY-MM-DD)의 뉴스 브리핑과 영상별 요약(채널, 제목, video_id, 핵심 내용)을 반환한다.
    그날 뉴스에 대한 질문에는 가장 먼저 이 도구를 사용한다."""
    day = load_day(date)
    if day is None:
        return f"{date} 데이터가 없습니다. 저장된 날짜: {', '.join(list_days()) or '없음'}"

    lines = [f"# {date} 브리핑\n", day["digest"], "\n# 영상별 요약"]
    for s in day["summaries"]:
        points = "\n".join(f"  - {p}" for p in s["key_points"])
        lines.append(
            f"\n## [{s['channel']}] {s['title']}\n"
            f"video_id: {s['video_id']} | URL: {s['url']}\n"
            f"헤드라인: {s['headline']}\n{points}"
        )
    return "\n".join(lines)


def search_transcripts(date: str, keyword: str) -> str:
    """해당 날짜 영상들의 자막 원문에서 keyword(종목명, 인물, 지표 등)가 나오는 부분을 찾아
    앞뒤 문맥과 함께 반환한다. 요약에 없는 세부 내용을 확인할 때 사용한다."""
    day = load_day(date)
    if day is None:
        return f"{date} 데이터가 없습니다."

    results = []
    for s in day["summaries"]:
        text = get_cached_transcript(s["video_id"])
        if not text:
            continue
        start, found = 0, 0
        while found < MAX_SNIPPETS_PER_VIDEO and (i := text.find(keyword, start)) != -1:
            snippet = text[max(0, i - SNIPPET_RADIUS): i + len(keyword) + SNIPPET_RADIUS]
            results.append(f"[{s['channel']}] {s['title']} (video_id: {s['video_id']}, 위치 {i})\n…{snippet}…")
            start, found = i + len(keyword), found + 1
    return "\n\n".join(results) or f"'{keyword}'가 {date} 자막에서 발견되지 않았습니다. 다른 표기로 검색해 보세요."


def read_transcript(video_id: str, start_char: int = 0) -> str:
    """영상 자막 원문을 start_char 위치부터 약 6000자 읽는다. 발언의 전체 맥락이 필요할 때 사용한다.
    결과 끝에 표시된 다음 위치로 이어서 읽을 수 있다."""
    text = get_cached_transcript(video_id)
    if not text:
        return f"{video_id} 자막이 저장되어 있지 않습니다."
    chunk = text[start_char: start_char + READ_CHUNK_CHARS]
    end = start_char + len(chunk)
    tail = f"\n\n(다음 위치: {end})" if end < len(text) else "\n\n(끝)"
    return f"[{start_char}~{end} / 전체 {len(text)}자]\n{chunk}{tail}"


# ---------- 에이전트 ----------

@dynamic_prompt
def _system_prompt(request: ModelRequest) -> str:
    days = list_days()
    latest = days[0] if days else "없음"
    return f"""당신은 한국 금융·경제 유튜브 방송을 매일 정리해 온 시장 브리핑 어시스턴트입니다.
오늘 날짜(KST): {datetime.now(KST).date().isoformat()}
저장된 가장 최근 날짜: {latest}

답변 원칙:
- 사용자가 날짜를 말하지 않으면 가장 최근 날짜를 기준으로 답하고, 어느 날짜 기준인지 밝힙니다.
- 먼저 get_day_overview로 그날 브리핑을 보고, 세부 내용은 search_transcripts와 read_transcript로 자막에서 확인합니다.
- 도구로 확인한 내용만 답하고, 없는 내용은 "방송에서 다루지 않았다"고 말합니다. 추측하지 않습니다.
- 사실(발표된 지표, 실적, 정책)과 출연자 개인의 견해를 구분하고, 견해는 누구의 것인지 밝힙니다.
- 채널 간 수치나 시각이 다르면 그 차이를 알려줍니다.
- 자막 원문을 길게 옮겨 적지 말고 핵심만 정리하며, 출처를 [채널명](URL) 형식으로 답니다.
- 투자 권유가 아니라 방송 내용 정리라는 점을 유지합니다.
- 한국어로 간결하게 답합니다."""


graph = create_agent(
    model=ChatOpenAI(model=CHAT_MODEL),
    tools=[list_available_days, get_day_overview, search_transcripts, read_transcript],
    middleware=[_system_prompt],
)
