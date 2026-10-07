# src/youtube.py
import glob
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
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
    MAX_VIDEO_SECONDS,
    MAX_VIDEOS_PER_CHANNEL,
    MAX_VIDEOS_PER_CHANNEL_LIMIT,
    VIDEOS_PER_DAY_BACK,
    TRANSCRIPT_DIR,
    UPLOAD_TIMES_PATH,
)


# 이보다 오래된 영상은 목록에 '1 day ago'처럼 일 단위로만 표시된다
COARSE_TIMESTAMP_AGE_SECONDS = 20 * 3600


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


def _list_depth(target: date) -> int:
    """지난 날짜일수록 더 많은 최신 영상을 훑는다 (하루 15~20개씩 올리는 채널 대비)."""
    days_back = max((datetime.now(KST).date() - target).days, 0)
    return min(MAX_VIDEOS_PER_CHANNEL + VIDEOS_PER_DAY_BACK * days_back, MAX_VIDEOS_PER_CHANNEL_LIMIT)


def _list_entries(handle: str, depth: int, lang: str | None = None) -> list[dict]:
    extractor_args = {"youtubetab": {"approximate_date": ["true"]}}
    if lang:
        extractor_args["youtube"] = {"lang": [lang]}
    opts = {
        "extract_flat": True,
        "playlistend": depth,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": extractor_args,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/{handle}/videos", download=False)
    return info.get("entries") or []


def _load_upload_times() -> dict[str, int]:
    path = Path(UPLOAD_TIMES_PATH)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save_upload_times(times: dict[str, int]) -> None:
    path = Path(UPLOAD_TIMES_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(times), encoding="utf-8")


def _exact_timestamp(video_id: str) -> int | None:
    """영상 페이지에서 정확한 업로드 시각(Unix 초)을 가져온다.

    요청이 몰리면 YouTube가 '봇이 아닌지 확인' 로그인을 요구해 실패할 수 있다 → None.
    """
    opts = {"skip_download": True, "quiet": True, "no_warnings": True, "noprogress": True, "logger": _SilentLogger()}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False, process=False)
    except Exception:
        return None
    return info.get("timestamp") or info.get("release_timestamp")


class _SilentLogger:
    """봇 확인 실패 메시지가 진행 로그를 덮지 않도록 yt-dlp 출력을 숨긴다."""
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass


_TITLE_FULL_DATE = re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일")
_TITLE_MONTH_DAY = re.compile(r"(?<!\d)(\d{1,2})월\s*(\d{1,2})일")
_TITLE_YYMMDD = re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)")


def _title_date(title: str, year: int) -> date | None:
    """제목에 적힌 방송 날짜 ('2026년 10월 1일', '10월 1일', '261001'). 없으면 None."""
    candidates = []
    if m := _TITLE_FULL_DATE.search(title):
        candidates.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if m := _TITLE_MONTH_DAY.search(title):
        candidates.append((year, int(m.group(1)), int(m.group(2))))
    for m in _TITLE_YYMMDD.finditer(title):
        if int(m.group(1)) == year % 100:
            candidates.append((2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for y, mo, d in candidates:
        try:
            return date(y, mo, d)
        except ValueError:
            continue
    return None


def list_videos_on(handle: str, channel_name: str, target: date) -> list[dict]:
    """채널의 '동영상' 탭에서 target 날짜(KST)에 올라온 영상 목록을 반환한다.

    목록의 업로드 시각은 '3 hours ago' 같은 상대 표기로 계산된 근사값이다.
    하루가 지나면 '1 day ago'처럼 일 단위로만 표시되어 날짜가 하루까지 틀릴 수 있으므로,
    그런 영상 중 target 근처인 것은 영상 페이지에서 정확한 시각을 다시 가져온다.
    yt-dlp는 영어 상대 표기만 해석할 수 있으므로 날짜는 기본(영어) 조회에서,
    제목은 자동 번역되지 않은 원문을 얻기 위해 한국어 조회에서 가져온다.
    """
    depth = _list_depth(target)
    entries = [
        e for e in _list_entries(handle, depth)
        if e.get("timestamp") is not None
        and e.get("live_status") not in ("is_live", "is_upcoming")
        and (e.get("duration") or 0) <= MAX_VIDEO_SECONDS  # 토큰 비용 때문에 너무 긴 영상 제외
    ]
    ko_titles = {e["id"]: e.get("title") for e in _list_entries(handle, depth, lang="ko")}

    # 일 단위 근사값('5 days ago')이고 target 앞뒤 하루 안에 있는 영상은 정확한 시각을 다시 조회.
    # 한 번 얻은 시각은 파일에 저장해 다시 요청하지 않는다 (요청이 몰리면 YouTube가 봇 확인을 요구함)
    now = datetime.now(KST).timestamp()
    coarse = {
        e["id"] for e in entries
        if now - e["timestamp"] >= COARSE_TIMESTAMP_AGE_SECONDS
        and abs((datetime.fromtimestamp(e["timestamp"], KST).date() - target).days) <= 1
    }
    known = _load_upload_times()
    missing = [vid for vid in coarse if vid not in known]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = dict(zip(missing, pool.map(_exact_timestamp, missing)))
    if any(fetched.values()):
        known.update({vid: ts for vid, ts in fetched.items() if ts})
        _save_upload_times(known)

    videos, dropped = [], 0
    for entry in entries:
        vid = entry["id"]
        title = ko_titles.get(vid) or entry.get("title") or ""
        estimated = False
        if vid not in coarse:
            published = datetime.fromtimestamp(entry["timestamp"], KST)
        elif vid in known:
            published = datetime.fromtimestamp(known[vid], KST)
        elif day := _title_date(title, target.year):
            # 정확한 시각을 못 얻으면 대략적인 날짜는 믿지 않고, 제목에 적힌 날짜를 쓴다 (시각은 모름 → 정오로 표기)
            published, estimated = datetime(day.year, day.month, day.day, 12, tzinfo=KST), True
        else:
            dropped += 1  # 날짜를 확정할 수 없어 후보에서 뺀다
            continue
        if published.date() != target:
            continue
        videos.append({
            "video_id": vid,
            "title": title,
            "channel": channel_name,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "published_at": published.isoformat(),
            "published_estimated": estimated,
            "duration": entry.get("duration"),
        })
    if dropped:
        print(f"    {channel_name}: 업로드 날짜를 확정하지 못한 영상 {dropped}개는 후보에서 제외", flush=True)
    return videos




_direct_blocked = False  # 직접 요청이 한 번 차단되면 이번 실행 동안은 Apify만 사용


def _fetch_direct(video_id: str) -> list[dict]:
    fetched = _transcript_api.fetch(video_id, languages=["ko"])
    return [{"start": s.start, "duration": s.duration, "text": s.text.strip()} for s in fetched if s.text.strip()]


def _apify_monthly_usage(token: str) -> float:
    response = httpx.get(
        "https://api.apify.com/v2/users/me/limits",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["data"]["current"]["monthlyUsageUsd"]


def _fetch_via_apify(video_id: str, token: str) -> list[dict] | None:
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
    raw = items[0].get("data") if items and isinstance(items[0], dict) else None
    if not isinstance(raw, list):
        return None
    return [
        {"start": float(s.get("start", 0)), "duration": float(s.get("dur", 0)), "text": s["text"].strip()}
        for s in raw if s.get("text", "").strip()
    ]


_NOT_FILENAME_CHAR = re.compile(r"[^0-9A-Za-z가-힣\[\]-]+")  # 한글·영문·숫자·[ ] - 만 남긴다
_TITLE_SEPARATORS = re.compile(r"[|｜ㅣ]")  # 이 뒤는 보통 출연자·코너명
TITLE_IN_FILENAME = 30


def _clean(text: str) -> str:
    return _NOT_FILENAME_CHAR.sub("_", text).strip("_")


def _file_stem(video: dict) -> str:
    """'날짜_채널명_제목요약_영상ID' 형식의 파일 이름 (확장자 제외)."""
    title = _clean(_TITLE_SEPARATORS.split(video.get("title") or "")[0])[:TITLE_IN_FILENAME].strip("_")
    channel = _clean(video.get("channel") or "")
    day = (video.get("published_at") or "")[:10]
    return "_".join(p for p in (day, channel, title, video["video_id"]) if p)


def _find(video_id: str, suffix: str) -> Path | None:
    """영상 ID로 저장된 파일을 찾는다 (새 이름 '…_영상ID.json'과 예전 이름 '영상ID.json' 모두)."""
    base = Path(TRANSCRIPT_DIR)
    legacy = base / f"{video_id}{suffix}"
    if legacy.exists():
        return legacy
    return next(base.glob(f"*_{glob.escape(video_id)}{suffix}"), None)


def _paths(video: dict) -> tuple[Path, Path]:
    """영상의 자막 파일 경로 (json, txt). 예전 이름으로 저장된 파일은 새 이름으로 바꾼다."""
    base = Path(TRANSCRIPT_DIR)
    stem = _file_stem(video)
    target = base / f"{stem}.json", base / f"{stem}.txt"
    for new, suffix in zip(target, (".json", ".txt")):
        old = _find(video["video_id"], suffix)
        if old and old != new:
            old.rename(new)
    return target


def _join(segments: list[dict]) -> str:
    return " ".join(s["text"] for s in segments)


def get_cached_transcript(video_id: str) -> str | None:
    """저장된 자막만 읽는다. 네트워크 요청이나 비용이 발생하지 않는다."""
    txt = _find(video_id, ".txt")
    return txt.read_text(encoding="utf-8") if txt else None


def get_segments(video: dict) -> list[dict] | None:
    """타임스탬프가 붙은 자막 구간 [{start, duration, text}, ...]을 반환한다.

    저장본(json) → 직접 요청(무료) → 차단 시 Apify(APIFY_API_TOKEN 설정 시) 순으로 시도하고,
    받은 구간은 data/transcripts/날짜_채널명_제목요약_영상ID.json (+ 이어 붙인 .txt)에 저장한다.
    예전 형식(.txt만 있음)은 타임스탬프가 없어 다시 받는다.
    자막이 없으면 None, 차단됐는데 Apify도 쓸 수 없으면 TranscriptBlockedError.
    """
    global _direct_blocked
    video_id = video["video_id"]
    seg_path, txt_path = _paths(video)
    if seg_path.exists():
        return json.loads(seg_path.read_text(encoding="utf-8"))

    segments = None
    if not _direct_blocked:
        try:
            segments = _fetch_direct(video_id)
        except (IpBlocked, RequestBlocked):
            _direct_blocked = True
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            return None

    if _direct_blocked:
        token = os.getenv("APIFY_API_TOKEN")
        if not token:
            raise TranscriptBlockedError(video_id)
        segments = _fetch_via_apify(video_id, token)

    if not segments:
        return None
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(_join(segments), encoding="utf-8")
    return segments


def get_transcript(video: dict) -> str | None:
    """자막 전체를 하나의 문자열로 반환한다 (요약용). 예전 형식 .txt가 있으면 다시 받지 않고 그대로 쓴다."""
    _, txt_path = _paths(video)
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")
    segments = get_segments(video)
    return _join(segments) if segments else None
