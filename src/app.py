# src/app.py
from langgraph.graph import END, START, StateGraph

from src.nodes import collect_facts, fetch_videos, index_transcripts, summarize_video, write_digest
from src.routers import route_videos
from src.state import NewsState

builder = StateGraph(NewsState)

builder.add_node("fetch_videos", fetch_videos)
builder.add_node("collect_facts", collect_facts)
builder.add_node("summarize_video", summarize_video)
builder.add_node("write_digest", write_digest)
builder.add_node("index_transcripts", index_transcripts)

builder.add_edge(START, "fetch_videos")
builder.add_edge("fetch_videos", "collect_facts")
builder.add_conditional_edges("collect_facts", route_videos, ["summarize_video", "write_digest"])
builder.add_edge("summarize_video", "write_digest")
builder.add_edge("write_digest", "index_transcripts")
builder.add_edge("index_transcripts", END)

graph = builder.compile()
