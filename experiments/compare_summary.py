# experiments/compare_summary.py
"""영상별 요약 한도 실험: A안(전체 한 번에) vs B안(타임스탬프 구간별 요약 → 합치기).

기존 코드는 바꾸지 않고 src의 프롬프트·스키마를 그대로 가져다 쓴다. 자막 원문은 출력하지 않는다.
사용법: uv run python -m experiments.compare_summary
결과: data/experiments/compare_summary.json
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

import numpy as np  # noqa: E402
import yt_dlp  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from src.config import SUMMARY_MODEL  # noqa: E402
from src.nodes import _SUMMARY_PROMPT, VideoSummary  # noqa: E402
from src.vectorstore import _embed, mmss  # noqa: E402
from src.youtube import _fetch_direct, _fetch_via_apify, _find  # noqa: E402

OUT_DIR = Path("data/experiments")
VIDEO_IDS = ["exEs4GMv-5o", "XCLV-Usem6U"]  # 9/30 삼프로TV 월가 뉴스레터(22k자), 마감시황(36k자)
A_MAX_CHARS = 70_000
B_SECTION_CHARS = 12_000
LATE_FROM = 20_000  # 기존 요약 한도. 이 뒤에서 나온 내용을 '후반부'로 본다
MATCH_WINDOW = 600  # 핵심 항목의 출처 위치를 찾을 때 비교하는 자막 구간 길이

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


# ---------- 자막 ----------

def load_video(video_id: str) -> dict:
    """영상 정보와 타임스탬프 자막. 기존 저장본 → 실험 폴더 → 새로 받기 순 (새로 받은 것은 실험 폴더에만 저장)."""
    with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False, process=False)
    video = {"video_id": video_id, "title": info.get("title", ""), "channel": info.get("channel", "")}

    saved = _find(video_id, ".json")
    cache = OUT_DIR / "transcripts" / f"{video_id}.json"
    if saved:
        segments = json.loads(saved.read_text(encoding="utf-8"))
    elif cache.exists():
        segments = json.loads(cache.read_text(encoding="utf-8"))
    else:
        try:
            segments = _fetch_direct(video_id)
        except Exception:
            import os
            segments = _fetch_via_apify(video_id, os.environ["APIFY_API_TOKEN"])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    return {**video, "segments": segments}


def text_with_offsets(segments: list[dict]) -> tuple[str, list[int]]:
    """이어 붙인 자막과 각 구간의 시작 글자 위치."""
    offsets, parts, pos = [], [], 0
    for s in segments:
        offsets.append(pos)
        parts.append(s["text"])
        pos += len(s["text"]) + 1
    return " ".join(parts), offsets


def sections(segments: list[dict], size: int) -> list[dict]:
    """타임스탬프(구간 경계) 기준으로 약 size자씩 나눈다."""
    out, cur, chars = [], [], 0
    for s in segments:
        if cur and chars + len(s["text"]) > size:
            out.append(cur)
            cur, chars = [], 0
        cur.append(s)
        chars += len(s["text"]) + 1
    if cur:
        out.append(cur)
    return [{"start": sec[0]["start"], "text": " ".join(s["text"] for s in sec)} for sec in out]


# ---------- LLM 호출 (토큰 집계) ----------

def call(prompt: str) -> tuple[VideoSummary, int, int]:
    llm = ChatOpenAI(model=SUMMARY_MODEL).with_structured_output(VideoSummary, include_raw=True)
    out = llm.invoke(prompt)
    usage = out["raw"].usage_metadata or {}
    return out["parsed"], usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def plan_a(video: dict, text: str) -> dict:
    t = time.time()
    summary, tin, tout = call(_SUMMARY_PROMPT.format(channel=video["channel"], title=video["title"], transcript=text[:A_MAX_CHARS]))
    return {"summary": summary, "calls": 1, "input_tokens": tin, "output_tokens": tout, "seconds": time.time() - t}


def plan_b(video: dict) -> dict:
    t = time.time()
    secs = sections(video["segments"], B_SECTION_CHARS)
    prompts = [
        _SUMMARY_PROMPT.format(channel=video["channel"], title=f"{video['title']} (구간 {mmss(s['start'])}부터)", transcript=s["text"])
        for s in secs
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:  # 구간 요약은 병렬 (실제 파이프라인처럼)
        parts = list(pool.map(call, prompts))
    section_text = "\n\n".join(
        f"[구간 시작 {mmss(s['start'])}]\n헤드라인: {p.headline}\n" + "\n".join(f"- {k}" for k in p.key_points)
        for s, (p, _, _) in zip(secs, parts)
    )
    merged, tin, tout = call(_MERGE_PROMPT.format(channel=video["channel"], title=video["title"], sections=section_text))
    return {
        "summary": merged,
        "calls": len(secs) + 1,
        "sections": len(secs),
        "input_tokens": sum(p[1] for p in parts) + tin,
        "output_tokens": sum(p[2] for p in parts) + tout,
        "seconds": time.time() - t,
    }


# ---------- 후반부 반영 판정 ----------

def late_points(points: list[str], text: str) -> list[dict]:
    """핵심 항목마다 가장 비슷한 자막 위치(글자 오프셋)를 bge-m3로 찾는다."""
    windows = list(range(0, max(len(text) - MATCH_WINDOW, 1), MATCH_WINDOW // 2))
    win_vecs = np.array(_embed([text[w: w + MATCH_WINDOW] for w in windows]))
    pt_vecs = np.array(_embed(points))
    best = (pt_vecs @ win_vecs.T).argmax(axis=1)
    return [{"point": p, "offset": windows[b], "late": windows[b] >= LATE_FROM} for p, b in zip(points, best)]


def main() -> None:
    results = []
    for vid in VIDEO_IDS:
        video = load_video(vid)
        text, _ = text_with_offsets(video["segments"])
        duration = video["segments"][-1]["start"]
        print(f"\n### {video['channel']} | {video['title'][:50]} | {len(text):,}자 | {mmss(duration)}", flush=True)
        row = {"video_id": vid, "title": video["title"], "chars": len(text), "late_chars": max(len(text) - LATE_FROM, 0)}
        for name, run in (("A", lambda: plan_a(video, text)), ("B", lambda: plan_b(video))):
            r = run()
            matches = late_points(r["summary"].key_points, text)
            r["late_count"] = sum(m["late"] for m in matches)
            r["matches"] = [{"offset": m["offset"], "late": m["late"]} for m in matches]
            r["summary"] = r["summary"].model_dump()
            row[name] = r
            print(f"  {name}안: 호출 {r['calls']} | 입력 {r['input_tokens']:,} | 출력 {r['output_tokens']:,} | "
                  f"{r['seconds']:.1f}초 | 후반부 핵심 {r['late_count']}/{len(matches)}", flush=True)
        results.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "compare_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT_DIR / 'compare_summary.json'}")


if __name__ == "__main__":
    main()
