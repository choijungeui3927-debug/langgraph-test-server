# src/state.py
import operator
from typing import Annotated, TypedDict


class NewsState(TypedDict, total=False):
    # 입력 (모두 선택): 비우면 오늘 날짜(KST)와 config.CHANNELS 사용
    target_date: str  # "YYYY-MM-DD"
    channels: dict[str, str]  # {"@handle": "표시 이름"}

    # 중간 결과
    videos: list[dict]
    summaries: Annotated[list[dict], operator.add]  # 병렬 요약 결과가 누적됨
    skipped: Annotated[list[dict], operator.add]  # 자막 없음 등으로 제외된 영상

    # 최종 결과
    digest: str


class VideoState(TypedDict):
    """summarize_video 노드 하나에 전달되는 개별 영상 상태."""
    video: dict
