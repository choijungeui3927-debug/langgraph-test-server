# src/nodes.py
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import (
    CHANNELS,
    CLOSING_KEYWORDS,
    DIGEST_MODEL,
    KST,
    MIN_TRANSCRIPT_CHARS,
    SINGLE_PASS_MAX_CHARS,
    SELECT_MODEL,
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
from src.vectorstore import index_video, is_indexed, mmss, remove_other_videos
from src.youtube import (
    ApifyBudgetExceededError,
    TranscriptBlockedError,
    get_segments,
    get_transcript,
    list_videos_on,
)


def _log(message: str) -> None:
    """진행 상황을 터미널에 바로 보여준다 (실행이 1~2분 걸려 멈춘 것처럼 보이지 않도록)."""
    print(f"[{datetime.now(KST):%H:%M:%S}] {message}", flush=True)


# ---------- 1. 영상 수집 ----------

def fetch_videos(state: NewsState) -> dict:
    target_str = state.get("target_date") or datetime.now(KST).date().isoformat()
    target = date.fromisoformat(target_str)
    channels = state.get("channels") or CHANNELS

    _log(f"1/5 영상 목록 수집 ({target_str}, 채널 {len(channels)}곳)")
    per_channel = []
    for handle, name in channels.items():
        try:
            per_channel.append(list_videos_on(handle, name, target))
            _log(f"    {name}: 후보 {len(per_channel[-1])}개")
        except Exception as e:
            _log(f"    {name}({handle}) 목록 수집 실패: {e}")

    videos = _select_videos(per_channel, target_str)
    _log(f"    선정 {len(videos)}개")
    for v in videos:
        _log(f"    - [{v.get('segment') or '일반'}] {v['channel']} {v['title'][:40]}")
    return {"target_date": target_str, "videos": videos}


# ---------- 1-2. 확정 숫자 수집 (요약 전에) ----------

def collect_facts(state: NewsState) -> dict:
    """자막에서 언급된 종목을 뽑고, 지수·수급·환율·종목 등락률을 코드로 확정한다.

    여기서 받은 자막은 캐시에 저장되므로 다음 요약 단계에서 다시 요청하지 않는다.
    """
    _log("2/5 자막 수집 + 확정 숫자 수집 (KRX·ECOS)")
    texts = []
    for v in state.get("videos") or []:
        try:
            text = get_transcript(v) or ""
            texts.append(text)
            _log(f"    자막 {len(text):,}자 — {v['title'][:40]}")
        except (TranscriptBlockedError, ApifyBudgetExceededError, httpx.HTTPError) as e:
            _log(f"    자막 실패 ({type(e).__name__}) — {v['title'][:40]}")  # 사유는 summarize_video에서 기록

    target = date.fromisoformat(state["target_date"])
    try:
        stock_names = extract_stocks(texts, target, exclude=set(CHANNELS.values()))
    except Exception as e:
        _log(f"    종목 추출 실패: {e}")
        stock_names = []
    facts = get_market_facts(state["target_date"], stock_names)
    _log(f"    종목 {len(facts['stocks'])}개 등락률 확정" + (f", 수집 오류: {facts['errors']}" if facts["errors"] else ""))
    _log(f"3/5 영상 {len(state.get('videos') or [])}개 요약 (병렬)")
    return {"facts": facts}


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


class _Pick(BaseModel):
    index: int = Field(description="후보 번호")
    reason: str = Field(description="이 영상이 그날 시황을 잘 보여주는 이유 (한 문장)")


class _Picks(BaseModel):
    picks: list[_Pick]


_SELECT_PROMPT = """당신은 금융·경제 뉴스 에디터입니다. 아래는 {date}에 유튜브 경제 채널에 올라온 영상 후보입니다.
다음 채널마다 정확히 1개씩 고르세요: {channels}
고른 영상들을 합쳤을 때 그날 시황(당일 한국장, 간밤 미국장)을 가장 잘 보여주도록 고릅니다.

고르는 기준 (위가 우선):
1. 그날 장을 정리하는 시황 방송 (장 마감·클로징·한국시황, 장중 시황, 장 시작 전 시황)
2. 그날 시장을 움직인 핵심 이슈(지수, 수급, 주도 업종, 금리·환율, 주요 실적)를 다루는 분석
3. 채널끼리 내용이 겹치지 않고, 시간대(장 전·장중·장 마감 후)가 다양할 것

제외할 것: 개별 종목 상담·추천 위주, 세미나·행사 홍보, 가상자산·크립토, 시장과 무관한 주제(국방, 사회 사건 등), 1~3분짜리 단신
업로드 시각은 KST입니다. 한국장은 15:30에 끝나고, 미국장은 다음 날 새벽(KST)에 끝납니다.

이미 고른 영상 (후보가 하나뿐인 채널):
{picked}

후보:
{candidates}"""


def _describe(i: int | None, v: dict) -> str:
    num = f"{i}. " if i is not None else "- "
    seg = f" [{v['segment']}]" if v.get("segment") else ""
    minutes = round((v.get("duration") or 0) / 60)
    return f"{num}[{v['channel']}] {v['published_at'][11:16]} {minutes}분{seg} {v['title']}"


def _llm_pick_per_channel(candidates: list[dict], picked: list[dict], channels: list[str], target_date: str) -> list[dict]:
    """여러 채널의 후보를 한 번에 LLM에 보여주고 채널마다 1개씩 고른다."""
    llm = ChatOpenAI(model=SELECT_MODEL).with_structured_output(_Picks)
    result: _Picks = llm.invoke(_SELECT_PROMPT.format(
        date=target_date,
        channels=", ".join(channels),
        picked="\n".join(_describe(None, v) for v in picked) or "(없음)",
        candidates="\n".join(_describe(i, v) for i, v in enumerate(candidates)),
    ))
    chosen: dict[str, dict] = {}
    for p in result.picks:
        if 0 <= p.index < len(candidates):
            v = candidates[p.index]
            chosen.setdefault(v["channel"], {**v, "selection_reason": p.reason})  # 채널당 첫 선택만
    return list(chosen.values())


def _fallback_pick(videos: list[dict]) -> dict:
    """LLM을 쓸 수 없을 때: 한국장 마감 → 미국장 마감 → 가장 최근 영상."""
    for segment in ("한국장 마감", "미국장 마감"):
        if found := next((v for v in videos if v["segment"] == segment), None):
            return {**found, "selection_reason": f"대체 규칙: {segment} 방송"}
    return {**videos[0], "selection_reason": "대체 규칙: 가장 최근 영상"}


def _select_videos(per_channel: list[list[dict]], target_date: str) -> list[dict]:
    """채널마다 그날 시황을 가장 잘 보여주는 영상을 1개씩 고른다.

    1) 후보가 하나뿐인 채널은 그 영상을 쓴다
    2) 후보가 여러 개인 채널들은 LLM이 한 번에 보고 채널마다 1개씩 고른다
    3) LLM이 실패하거나 빠뜨린 채널은 한국장 마감 → 미국장 마감 → 최신순 규칙으로 채운다
    """
    groups = [g for g in per_channel if g]
    for g in groups:
        for v in g:
            v["segment"] = _closing_segment(v)

    picked = [{**g[0], "selection_reason": "그 채널의 그날 유일한 후보"} for g in groups if len(g) == 1]
    multi = [g for g in groups if len(g) > 1]
    if multi:
        candidates = [v for g in multi for v in g]
        try:
            picked += _llm_pick_per_channel(candidates, picked, [g[0]["channel"] for g in multi], target_date)
        except Exception as e:
            _log(f"    LLM 영상 선정 실패, 규칙으로 대체: {e}")
        done = {v["channel"] for v in picked}
        picked += [_fallback_pick(g) for g in multi if g[0]["channel"] not in done]

    order = {g[0]["channel"]: i for i, g in enumerate(groups)}  # 채널 설정 순서대로
    return sorted(picked, key=lambda v: order[v["channel"]])


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


_MERGE_PROMPT = """다음은 한 유튜브 금융·경제 방송을 시간 구간별로 나눠 요약한 결과입니다.
이를 합쳐 영상 전체의 요약 하나를 만드세요.

규칙:
- 같은 주장이나 같은 사실이 여러 구간에 나오면 한 번만 씁니다.
- 핵심(key_points)마다 그 내용이 나온 구간의 시작 시각을 앞에 [분:초]로 표기합니다.
- 영상 앞부분에 치우치지 말고 전체에서 중요한 내용을 고릅니다.
- 출연자의 의견과 사실을 구분하고, 의견이면 누구의 견해인지 밝힙니다.

채널: {channel}
제목: {title}

구간별 요약:
{sections}"""


def _summarize_once(video: dict, title: str, text: str) -> VideoSummary:
    llm = ChatOpenAI(model=SUMMARY_MODEL).with_structured_output(VideoSummary)
    return llm.invoke(_SUMMARY_PROMPT.format(channel=video["channel"], title=title, transcript=text))


def _sections(video: dict, transcript: str) -> list[dict]:
    """자막을 균등한 구간으로 나눈다 → [{start, text}].

    구간 수 = ceil(글자 수 / SINGLE_PASS_MAX_CHARS), 구간 크기 = 글자 수 / 구간 수 (예: 45,000자 → 15,000자 × 3).
    경계는 타임스탬프(자막 구간 경계)에 맞추고, 타임스탬프가 없는 예전 형식이면 글자 위치로 나누고 start는 None.
    """
    n = math.ceil(len(transcript) / SINGLE_PASS_MAX_CHARS)
    try:
        segments = get_segments(video)
    except Exception:
        segments = None
    if not segments:
        size = math.ceil(len(transcript) / n)
        return [{"start": None, "text": transcript[i: i + size]} for i in range(0, len(transcript), size)]

    total = sum(len(s["text"]) + 1 for s in segments)
    groups: list[list[dict]] = [[] for _ in range(n)]
    pos = 0
    for s in segments:
        mid = pos + len(s["text"]) / 2  # 구간 가운데 글자가 속한 쪽으로 배정해 경계 오차를 줄인다
        groups[min(int(mid * n / total), n - 1)].append(s)
        pos += len(s["text"]) + 1
    return [{"start": g[0]["start"], "text": " ".join(s["text"] for s in g)} for g in groups if g]


def _summarize_by_sections(video: dict, sections: list[dict]) -> VideoSummary:
    """구간별로 요약(병렬)한 뒤 합친다. 같은 주장은 한 번만, 핵심마다 구간 시작 시각을 붙인다."""
    def label(s: dict) -> str:
        return mmss(s["start"]) if s["start"] is not None else f"구간 {sections.index(s) + 1}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(
            lambda s: _summarize_once(video, f"{video['title']} ({label(s)}부터)", s["text"]), sections,
        ))
    merged = "\n\n".join(
        f"[구간 시작 {label(s)}]\n헤드라인: {p.headline}\n" + "\n".join(f"- {k}" for k in p.key_points)
        for s, p in zip(sections, parts)
    )
    llm = ChatOpenAI(model=SUMMARY_MODEL).with_structured_output(VideoSummary)
    return llm.invoke(_MERGE_PROMPT.format(channel=video["channel"], title=video["title"], sections=merged))


def summarize_video(state: VideoState) -> dict:
    video = state["video"]
    try:
        transcript = get_transcript(video)
    except TranscriptBlockedError:
        return {"skipped": [{**video, "reason": "유튜브 IP 차단으로 자막 수집 실패"}]}
    except ApifyBudgetExceededError:
        return {"skipped": [{**video, "reason": "Apify 월 예산 초과로 자막 수집 중단"}]}
    except httpx.HTTPError as e:
        return {"skipped": [{**video, "reason": f"Apify 자막 수집 실패 ({type(e).__name__})"}]}
    if not transcript or len(transcript) < MIN_TRANSCRIPT_CHARS:
        return {"skipped": [{**video, "reason": "자막 없음 또는 너무 짧음"}]}

    if len(transcript) <= SINGLE_PASS_MAX_CHARS:
        result = _summarize_once(video, video["title"], transcript)
        how = "한 번에"
    else:
        sections = _sections(video, transcript)
        result = _summarize_by_sections(video, sections)
        how = f"구간 {len(sections)}개"
    _log(f"    요약 완료 ({len(transcript):,}자, {how}) — {video['title'][:40]}")
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
    _log(f"4/5 브리핑 작성 ({DIGEST_MODEL})")
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
        _log(f"    숫자 검증: 오류 {len(fact_check['first_errors'])}건")
        if fact_check["first_errors"]:
            _log("    확정 숫자와 다른 부분이 있어 브리핑을 1회 다시 작성")
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
    _log(f"    저장 완료: data/{target_date}.md")
    return {"digest": digest, "fact_check": fact_check}


# ---------- 4. 채팅 검색용 자막 색인 ----------

def index_day(target_date: str, videos: list[dict], force: bool = False) -> int:
    """영상 자막 전체를 타임스탬프 청크로 나눠 벡터 DB(ChromaDB)에 저장한다. 저장한 청크 수를 반환.

    예전 형식(.txt) 자막은 타임스탬프가 없어 다시 받는다 (차단 시 Apify, 영상당 약 $0.01).
    """
    total = 0
    for v in videos:
        if not force and is_indexed(v["video_id"]):
            _log(f"    이미 색인됨 — {v['title'][:40]}")
            continue
        try:
            segments = get_segments(v)
        except (TranscriptBlockedError, ApifyBudgetExceededError, httpx.HTTPError) as e:
            _log(f"    자막 실패 ({type(e).__name__}) — {v['title'][:40]}")
            continue
        if not segments:
            _log(f"    자막 없음 — {v['title'][:40]}")
            continue
        n = index_video(v, target_date, segments)
        total += n
        _log(f"    청크 {n}개 저장 ({mmss(segments[-1]['start'])} 분량) — {v['title'][:40]}")
    return total


def index_transcripts(state: NewsState) -> dict:
    summaries = state.get("summaries") or []
    _log(f"5/5 채팅 검색용 자막 색인 (bge-m3 → ChromaDB, 영상 {len(summaries)}개)")
    removed = remove_other_videos(state["target_date"], {s["video_id"] for s in summaries})
    if removed:
        _log(f"    이번에 선정되지 않은 영상의 청크 {removed}개 삭제")
    index_day(state["target_date"], summaries)
    return {}
