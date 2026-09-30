# src/routers.py
from langgraph.types import Send

from src.state import NewsState


def route_videos(state: NewsState) -> list[Send] | str:
    """수집된 영상마다 summarize_video 노드를 병렬로 실행한다. 영상이 없으면 바로 종합 단계로."""
    videos = state.get("videos") or []
    if not videos:
        return "write_digest"
    return [Send("summarize_video", {"video": v}) for v in videos]
