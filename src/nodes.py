# src/nodes.py
from datetime import date, datetime
from itertools import zip_longest

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import (
    CHANNELS,
    DIGEST_MODEL,
    KST,
    MAX_TRANSCRIPT_CHARS,
    MAX_VIDEOS_TOTAL,
    MIN_TRANSCRIPT_CHARS,
    SUMMARY_MODEL,
)
from src.state import NewsState, VideoState
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
    return {"target_date": target_str, "videos": _pick_round_robin(per_channel, MAX_VIDEOS_TOTAL)}


def _pick_round_robin(per_channel: list[list[dict]], limit: int) -> list[dict]:
    """채널을 번갈아 가며 최신 영상부터 limit개를 고른다. (한 채널 쏠림 방지)"""
    picked = []
    for group in zip_longest(*per_channel):
        picked.extend(v for v in group if v is not None)
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
이를 바탕으로 한국어 '오늘의 금융·경제 뉴스 브리핑'을 마크다운으로 작성하세요.

규칙:
- 맨 위에 오늘 시장을 3줄 이내로 정리한 '핵심 요약'을 둡니다.
- 이어서 주제별 섹션(예: 국내 증시, 미국·해외 증시, 금리·환율·원자재, 산업·기업, 정책·규제, 부동산)으로 묶습니다. 내용이 없는 섹션은 생략합니다.
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
        f"주제: {', '.join(s['topics'])}\n"
        f"헤드라인: {s['headline']}\n"
        f"핵심:\n{points}"
    )


def write_digest(state: NewsState) -> dict:
    summaries = state.get("summaries") or []
    if not summaries:
        return {"digest": f"{state['target_date']}에 요약할 수 있는 뉴스 영상이 없습니다."}

    summaries = sorted(summaries, key=lambda s: s["published_at"])
    llm = ChatOpenAI(model=DIGEST_MODEL)
    response = llm.invoke(_DIGEST_PROMPT.format(
        date=state["target_date"],
        summaries="\n\n".join(_format_summary(s) for s in summaries),
    ))
    return {"digest": response.content}
