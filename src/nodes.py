# src/nodes.py
from datetime import date, datetime
from itertools import zip_longest

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import (
    CHANNELS,
    CLOSING_KEYWORDS,
    DIGEST_MODEL,
    KST,
    MAX_TRANSCRIPT_CHARS,
    MAX_VIDEOS_TOTAL,
    MIN_TRANSCRIPT_CHARS,
    SUMMARY_MODEL,
    US_CLOSE_UPLOAD_BEFORE_HOUR,
    US_MARKET_KEYWORDS,
)
from src.facts import (
    FACT_RULES,
    extract_stocks,
    facts_for_prompt,
    get_market_facts,
    render_numbers,
    verify,
)
from src.state import NewsState, VideoState
from src.storage import log_fact_errors, save_day
from src.youtube import (
    ApifyBudgetExceededError,
    TranscriptBlockedError,
    get_transcript,
    list_videos_on,
)


# ---------- 1. 영상 수집 ----------

def fetch_videos(state: NewsState) -> dict:
    target_str = state.get("target_date") or datetime.now(KST).date().isoformat()
    target = date.fromisoformat(target_str)
    channels = state.get("channels") or CHANNELS

    per_channel = []
    for handle, name in channels.items():
        try:
            per_channel.append(list_videos_on(handle, name, target))
        except Exception as e:
            print(f"[fetch_videos] {name}({handle}) 목록 수집 실패: {e}")
    return {"target_date": target_str, "videos": _select_videos(per_channel, MAX_VIDEOS_TOTAL)}


# ---------- 1-2. 확정 숫자 수집 (요약 전에) ----------

def collect_facts(state: NewsState) -> dict:
    """자막에서 언급된 종목을 뽑고, 지수·수급·환율·종목 등락률을 코드로 확정한다.

    여기서 받은 자막은 캐시에 저장되므로 다음 요약 단계에서 다시 요청하지 않는다.
    """
    texts = []
    for v in state.get("videos") or []:
        try:
            texts.append(get_transcript(v["video_id"]) or "")
        except (TranscriptBlockedError, ApifyBudgetExceededError, httpx.HTTPError):
            pass  # 수집 실패 사유는 summarize_video에서 기록한다

    target = date.fromisoformat(state["target_date"])
    try:
        stock_names = extract_stocks(texts, target, exclude=set(CHANNELS.values()))
    except Exception as e:
        print(f"[collect_facts] 종목 추출 실패: {e}")
        stock_names = []
    return {"facts": get_market_facts(state["target_date"], stock_names)}


def _closing_segment(video: dict) -> str | None:
    """마감 방송이면 어느 장의 마감인지 반환한다 ('한국장 마감' / '미국장 마감'), 아니면 None."""
    title = video["title"]
    is_us = any(k in title for k in US_MARKET_KEYWORDS)
    has_closing_word = any(k in title for k in CLOSING_KEYWORDS)
    uploaded_morning = datetime.fromisoformat(video["published_at"]).hour < US_CLOSE_UPLOAD_BEFORE_HOUR
    if is_us and (has_closing_word or uploaded_morning):
        return "미국장 마감"
    if has_closing_word:
        return "한국장 마감"
    return None


def _round_robin(per_channel: list[list[dict]]) -> list[dict]:
    """채널을 번갈아 가며 최신 영상부터 나열한다. (한 채널 쏠림 방지)"""
    ordered = []
    for group in zip_longest(*per_channel):
        ordered.extend(v for v in group if v is not None)
    return ordered


def _select_videos(per_channel: list[list[dict]], limit: int) -> list[dict]:
    """마감 방송을 우선해 limit개를 고른다.

    1) 한국장 마감 1개, 미국장 마감 1개를 먼저 (두 시장을 모두 담기 위해)
    2) 남은 마감 방송
    3) 일반 영상 (채널을 번갈아 최신순)
    """
    ordered = _round_robin(per_channel)
    for v in ordered:
        v["segment"] = _closing_segment(v)

    picked = []
    for segment in ("한국장 마감", "미국장 마감"):
        first = next((v for v in ordered if v["segment"] == segment), None)
        if first:
            picked.append(first)
    picked += [v for v in ordered if v["segment"] and v not in picked]
    picked += [v for v in ordered if not v["segment"]]
    return picked[:limit]


# ---------- 2. 영상별 요약 (병렬) ----------

class VideoSummary(BaseModel):
    headline: str = Field(description="영상 핵심을 한 문장으로 요약한 헤드라인")
    key_points: list[str] = Field(description="핵심 사실·수치·전망 3~6개. 구체적인 숫자와 고유명사를 유지")
    topics: list[str] = Field(description="관련 분야 태그 (예: 국내증시, 미국증시, 금리, 환율, 부동산, 반도체, 정책)")


_SUMMARY_PROMPT = """다음은 한국어 금융·경제 유튜브 영상의 자동 생성 자막입니다.
자막에는 오타·띄어쓰기 오류·화자 표시가 섞여 있을 수 있으니 문맥으로 보정해서 이해하세요.
출연자의 의견과 사실(발표된 지표, 실적, 정책)을 구분하고, 의견이면 누구의 견해인지 밝히세요.

채널: {channel}
제목: {title}

자막:
{transcript}"""


def summarize_video(state: VideoState) -> dict:
    video = state["video"]
    try:
        transcript = get_transcript(video["video_id"])
    except TranscriptBlockedError:
        return {"skipped": [{**video, "reason": "유튜브 IP 차단으로 자막 수집 실패"}]}
    except ApifyBudgetExceededError:
        return {"skipped": [{**video, "reason": "Apify 월 예산 초과로 자막 수집 중단"}]}
    except httpx.HTTPError as e:
        return {"skipped": [{**video, "reason": f"Apify 자막 수집 실패 ({type(e).__name__})"}]}
    if not transcript or len(transcript) < MIN_TRANSCRIPT_CHARS:
        return {"skipped": [{**video, "reason": "자막 없음 또는 너무 짧음"}]}

    llm = ChatOpenAI(model=SUMMARY_MODEL).with_structured_output(VideoSummary)
    result: VideoSummary = llm.invoke(_SUMMARY_PROMPT.format(
        channel=video["channel"],
        title=video["title"],
        transcript=transcript[:MAX_TRANSCRIPT_CHARS],
    ))
    return {"summaries": [{**video, **result.model_dump()}]}


# ---------- 3. 하루 뉴스 종합 ----------

_DIGEST_PROMPT = """당신은 금융·경제 뉴스 에디터입니다. 아래는 {date}에 여러 유튜브 경제 채널에 올라온 영상들의 요약입니다.
이를 바탕으로 한국어 금융·경제 뉴스 브리핑을 마크다운으로 작성하세요. 제목은 '# {date} 금융·경제 뉴스 브리핑'입니다.

{fact_rules}

확정 숫자:
{facts}

규칙:
- 제목 아래에 그날 시장을 3줄 이내로 정리한 '핵심 요약'을 둡니다.
- 이어서 주제별 섹션(예: 국내 증시, 미국·해외 증시, 금리·환율·원자재, 산업·기업, 정책·규제, 부동산)으로 묶습니다. 내용이 없는 섹션은 생략합니다.
- 한국장과 미국장을 섞지 않습니다. 한국장은 {date} 당일(15:30 KST 마감) 장이고,
  '미국장 마감' 방송은 {date} 오전(KST)에 올라온 것으로 미국 현지 기준 전날 장을 다룹니다.
  미국 증시 내용에는 "미국 현지 ○일 장" 또는 "간밤 뉴욕증시"처럼 어느 장인지 밝힙니다.
- '마감' 표시가 있는 방송의 지수·수급 수치를 우선 사용합니다.
- 여러 영상에서 같은 이슈를 다루면 하나로 합치고, 채널 간 시각이 다르면 그 차이를 짧게 적습니다.
- 수치·종목명·기관명은 요약에 있는 그대로 쓰고, 요약에 없는 내용은 추가하지 않습니다.
- 각 항목 끝에 출처를 [채널명](URL) 형식으로 답니다.
- 전문가 개인의 전망은 사실과 구분해 '~의 견해' 형태로 씁니다.

영상 요약:
{summaries}"""


def _format_summary(s: dict) -> str:
    points = "\n".join(f"  - {p}" for p in s["key_points"])
    return (
        f"### [{s['channel']}] {s['title']}\n"
        f"URL: {s['url']}\n"
        f"업로드: {s['published_at'][:16]} KST | 구분: {s.get('segment') or '일반'}\n"
        f"주제: {', '.join(s['topics'])}\n"
        f"헤드라인: {s['headline']}\n"
        f"핵심:\n{points}"
    )


_RETRY_PROMPT = """방금 작성한 브리핑에서 확정 숫자와 맞지 않는 부분이 발견됐습니다.
아래 오류를 모두 고친 브리핑 전체를 다시 작성하세요. 확정 숫자에 있는 항목은 확정 숫자만 사용하고,
확정 숫자와 다른 방송 수치는 지웁니다.

오류:
{errors}

이전 브리핑:
{digest}"""


def _insert_numbers(body: str, numbers: str) -> str:
    """LLM이 쓴 제목('# ...') 바로 아래에 코드가 만든 '오늘의 숫자' 섹션을 끼워 넣는다."""
    first, _, rest = body.lstrip().partition("\n")
    if first.startswith("# "):
        return f"{first}\n\n{numbers}\n\n{rest.lstrip()}"
    return f"{numbers}\n\n{body}"


def write_digest(state: NewsState) -> dict:
    target_date = state["target_date"]
    facts = state.get("facts") or {}
    summaries = state.get("summaries") or []
    if not summaries:
        return {"digest": f"{target_date}에 요약할 수 있는 뉴스 영상이 없습니다."}

    summaries = sorted(summaries, key=lambda s: s["published_at"])
    llm = ChatOpenAI(model=DIGEST_MODEL)
    prompt = _DIGEST_PROMPT.format(
        date=target_date,
        fact_rules=FACT_RULES,
        facts=facts_for_prompt(facts) if facts else "(수집 실패 — 방송 수치를 쓰되 출처를 답니다)",
        summaries="\n\n".join(_format_summary(s) for s in summaries),
    )
    body = llm.invoke(prompt).content

    # 숫자 검증: 오류가 있으면 1회 재생성, 그래도 남으면 로그에 기록
    fact_check = {"first_errors": [], "final_errors": [], "regenerated": False}
    if facts:
        fact_check["first_errors"] = verify(body, facts)
        if fact_check["first_errors"]:
            body = llm.invoke(_RETRY_PROMPT.format(
                errors="\n".join(f"- {e}" for e in fact_check["first_errors"]), digest=body,
            )).content
            fact_check["regenerated"] = True
            fact_check["final_errors"] = verify(body, facts)
            if fact_check["final_errors"]:
                log_fact_errors(target_date, fact_check["final_errors"])

    digest = _insert_numbers(body, render_numbers(facts)) if facts else body
    # 채팅 그래프에서 그날 뉴스를 다시 불러올 수 있도록 저장
    save_day(target_date, state.get("videos") or [], summaries, digest, facts, fact_check)
    return {"digest": digest, "fact_check": fact_check}
