# src/vectorstore.py
"""자막을 타임스탬프 단위 청크로 나눠 로컬 임베딩(bge-m3)으로 ChromaDB에 저장하고 검색한다.

채팅의 세부 질문은 여기서 검색한 청크를 근거로 답하고, 청크마다 youtu.be 타임스탬프 링크를 붙인다.
"""
from functools import lru_cache

from src.config import CHROMA_DIR, EMBEDDING_MODEL

COLLECTION = "transcripts"
CHUNK_MAX_CHARS = 600  # 청크 하나의 최대 글자 수 (약 340토큰, 30~70초 분량)
CHUNK_MAX_SECONDS = 90  # 말이 느려도 청크가 이 시간을 넘지 않게
CHUNK_OVERLAP_SEGMENTS = 2  # 문맥이 끊기지 않도록 앞 청크의 마지막 구간을 다음 청크에 겹친다


@lru_cache(maxsize=1)
def _model():
    """bge-m3는 처음 불러올 때 수십 초 걸리므로 한 번만 불러온다."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _collection():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def _embed(texts: list[str]) -> list[list[float]]:
    return _model().encode(texts, batch_size=16, normalize_embeddings=True).tolist()


def timestamp_url(video_id: str, seconds: float) -> str:
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def chunk_segments(segments: list[dict]) -> list[dict]:
    """자막 구간을 이어 붙여 [{start, end, text}] 청크를 만든다. 청크 경계는 항상 구간 경계(타임스탬프)다."""
    chunks, i = [], 0
    while i < len(segments):
        j, chars = i, 0
        while j < len(segments):
            chars += len(segments[j]["text"]) + 1
            too_long = chars > CHUNK_MAX_CHARS or segments[j]["start"] - segments[i]["start"] > CHUNK_MAX_SECONDS
            if too_long and j > i:
                break
            j += 1
        last = segments[j - 1]
        chunks.append({
            "start": segments[i]["start"],
            "end": last["start"] + last.get("duration", 0),
            "text": " ".join(s["text"] for s in segments[i:j]),
        })
        if j >= len(segments):
            break
        i = max(i + 1, j - CHUNK_OVERLAP_SEGMENTS)
    return chunks


def remove_other_videos(target_date: str, keep_video_ids: set[str]) -> int:
    """그날 청크 중 이번에 선정되지 않은 영상의 청크를 지운다 (브리핑을 다시 만들 때). 지운 청크 수를 반환."""
    found = _collection().get(where={"date": target_date}, include=["metadatas"])
    stale = [i for i, m in zip(found["ids"], found["metadatas"]) if m["video_id"] not in keep_video_ids]
    if stale:
        _collection().delete(ids=stale)
    return len(stale)


def is_indexed(video_id: str) -> bool:
    return bool(_collection().get(where={"video_id": video_id}, limit=1)["ids"])


def index_video(video: dict, target_date: str, segments: list[dict]) -> int:
    """영상 하나의 자막을 청크로 나눠 저장한다. 같은 영상을 다시 넣으면 덮어쓴다. 저장한 청크 수를 반환."""
    chunks = chunk_segments(segments)
    if not chunks:
        return 0
    vid = video["video_id"]
    _collection().upsert(
        ids=[f"{vid}:{i:04d}" for i in range(len(chunks))],
        embeddings=_embed([c["text"] for c in chunks]),
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "video_id": vid,
            "title": video["title"],
            "channel": video["channel"],
            "date": target_date,
            "segment": video.get("segment") or "일반",
            "start": float(c["start"]),
            "end": float(c["end"]),
            "url": timestamp_url(vid, c["start"]),
        } for c in chunks],
    )
    return len(chunks)


def search(query: str, date: str | None = None, k: int = 8) -> list[dict]:
    """질문과 가까운 청크 k개를 반환한다. date를 주면 그날 영상에서만 찾는다."""
    result = _collection().query(
        query_embeddings=_embed([query]),
        n_results=k,
        where={"date": date} if date else None,
    )
    return [
        {**meta, "text": doc, "distance": dist}
        for meta, doc, dist in zip(result["metadatas"][0], result["documents"][0], result["distances"][0])
    ]
