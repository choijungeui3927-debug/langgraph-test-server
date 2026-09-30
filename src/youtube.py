# src/youtube.py
import os
from datetime import date, datetime
from pathlib import Path

import httpx
import yt_dlp
from youtube_transcript_api import (
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)
from youtube_transcript_api.proxies import GenericProxyConfig

from src.config import (
    APIFY_MONTHLY_BUDGET_USD,
    APIFY_TRANSCRIPT_ACTOR,
    KST,
    MAX_VIDEOS_PER_CHANNEL,
    TRANSCRIPT_CACHE_DIR,
)


class TranscriptBlockedError(Exception):
    """유튜브가 현재 IP의 자막 요청을 차단함 (보통 일시적)."""


class ApifyBudgetExceededError(Exception):
    """이번 달 Apify 사용액이 APIFY_MONTHLY_BUDGET_USD에 도달함."""


def _make_transcript_api() -> YouTubeTranscriptApi:
    # 유튜브 IP 차단 회피용 프록시 (선택): YOUTUBE_PROXY_URL=http://user:pass@host:port
    proxy_url = os.getenv("YOUTUBE_PROXY_URL")
    if proxy_url:
        return YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=proxy_url, https_url=proxy_url))
    return YouTubeTranscriptApi()


_transcript_api = _make_transcript_api()


def _list_entries(handle: str, lang: str | None = None) -> list[dict]:
    extractor_args = {"youtubetab": {"approximate_date": ["true"]}}
    if lang:
        extractor_args["youtube"] = {"lang": [lang]}
    opts = {
        "extract_flat": True,
        "playlistend": MAX_VIDEOS_PER_CHANNEL,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": extractor_args,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/{handle}/videos", download=False)
    return info.get("entries") or []


def list_videos_on(handle: str, channel_name: str, target: date) -> list[dict]:
    """채널의 '동영상' 탭에서 target 날짜(KST)에 올라온 영상 목록을 반환한다.

    업로드 시각은 '3 hours ago' 같은 상대 표기로 계산된 근사값이라 시간 단위 오차가 있다.
    yt-dlp는 영어 상대 표기만 해석할 수 있으므로 날짜는 기본(영어) 조회에서,
    제목은 자동 번역되지 않은 원문을 얻기 위해 한국어 조회에서 가져온다.
    """
    entries = _list_entries(handle)
    ko_titles = {e["id"]: e.get("title") for e in _list_entries(handle, lang="ko")}

    videos = []
    for entry in entries:
        ts = entry.get("timestamp")
        if ts is None or entry.get("live_status") in ("is_live", "is_upcoming"):
            continue
        published = datetime.fromtimestamp(ts, KST)
        if published.date() != target:
            continue
        videos.append({
            "video_id": entry["id"],
            "title": ko_titles.get(entry["id"]) or entry.get("title") or "",
            "channel": channel_name,
            "url": f"https://www.youtube.com/watch?v={entry['id']}",
            "published_at": published.isoformat(),
            "duration": entry.get("duration"),
        })
    return videos


_direct_blocked = False  # 직접 요청이 한 번 차단되면 이번 실행 동안은 Apify만 사용


def _fetch_direct(video_id: str) -> str | None:
    fetched = _transcript_api.fetch(video_id, languages=["ko"])
    return " ".join(s.text.strip() for s in fetched if s.text.strip())


def _apify_monthly_usage(token: str) -> float:
    response = httpx.get(
        "https://api.apify.com/v2/users/me/limits",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"]["current"]["monthlyUsageUsd"]


def _fetch_via_apify(video_id: str, token: str) -> str | None:
    """Apify Actor로 자막을 가져온다. Apify 서버에서 실행되므로 이 PC의 IP 차단과 무관하다.
    요금: 결과 1건당 약 $0.01. 이번 달 사용액이 예산을 넘으면 호출하지 않는다."""
    if _apify_monthly_usage(token) >= APIFY_MONTHLY_BUDGET_USD:
        raise ApifyBudgetExceededError(video_id)

    response = httpx.post(
        f"https://api.apify.com/v2/acts/{APIFY_TRANSCRIPT_ACTOR}/run-sync-get-dataset-items",
        headers={"Authorization": f"Bearer {token}"},
        params={"timeout": 240, "memory": 1024, "maxTotalChargeUsd": 0.01},
        json={"videoUrl": f"https://www.youtube.com/watch?v={video_id}", "targetLanguage": "ko"},
        timeout=270,
    )
    response.raise_for_status()
    items = response.json()
    segments = items[0].get("data") if items and isinstance(items[0], dict) else None
    if not isinstance(segments, list):
        return None
    return " ".join(s["text"].strip() for s in segments if s.get("text", "").strip())


def get_cached_transcript(video_id: str) -> str | None:
    """캐시에 저장된 자막만 읽는다. 네트워크 요청이나 비용이 발생하지 않는다."""
    cache = Path(TRANSCRIPT_CACHE_DIR) / f"{video_id}.txt"
    return cache.read_text(encoding="utf-8") if cache.exists() else None


def get_transcript(video_id: str) -> str | None:
    """한국어 자막(수동 우선, 없으면 자동 생성)을 하나의 문자열로 반환한다.

    캐시 → 직접 요청(무료) → 차단 시 Apify(APIFY_API_TOKEN 설정 시) 순으로 시도한다.
    자막이 없으면 None, 차단됐는데 Apify도 쓸 수 없으면 TranscriptBlockedError.
    """
    global _direct_blocked
    cache = Path(TRANSCRIPT_CACHE_DIR) / f"{video_id}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    text = None
    if not _direct_blocked:
        try:
            text = _fetch_direct(video_id)
        except (IpBlocked, RequestBlocked):
            _direct_blocked = True
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            return None

    if _direct_blocked:
        token = os.getenv("APIFY_API_TOKEN")
        if not token:
            raise TranscriptBlockedError(video_id)
        text = _fetch_via_apify(video_id, token)

    if not text:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text
