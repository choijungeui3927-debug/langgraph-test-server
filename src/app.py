# src/app.py
from langgraph.graph import END, START, StateGraph

from src.nodes import fetch_videos, summarize_video, write_digest
from src.routers import route_videos
from src.state import NewsState

builder = StateGraph(NewsState)

builder.add_node("fetch_videos", fetch_videos)
builder.add_node("summarize_video", summarize_video)
builder.add_node("write_digest", write_digest)

builder.add_edge(START, "fetch_videos")
builder.add_conditional_edges("fetch_videos", route_videos, ["summarize_video", "write_digest"])
builder.add_edge("summarize_video", "write_digest")
builder.add_edge("write_digest", END)

graph = builder.compile()
