# YouTube 금융·경제 뉴스 일일 브리핑

한국 금융·경제 유튜브 채널에 그날 올라온 영상의 자막을 모아 요약하고, 주제별 브리핑을 만든 뒤 그 내용으로 채팅할 수 있는 LangGraph 프로젝트입니다.

## 구성

| 그래프 | 역할 |
|---|---|
| `agent` ([src/app.py](src/app.py)) | 영상 수집 → 영상별 요약(병렬) → 브리핑 작성 후 `data/날짜.json`에 저장 |
| `chat` ([src/chat.py](src/chat.py)) | 저장된 브리핑·요약·자막 원문을 도구로 찾아보며 그날 시장에 대해 대화 |

```
fetch_videos → summarize_video (영상마다 병렬) → write_digest
```

| 파일 | 내용 |
|---|---|
| [src/config.py](src/config.py) | 채널 목록, 영상 수, 모델, Apify 예산 등 설정 |
| [src/youtube.py](src/youtube.py) | yt-dlp로 영상 목록 수집, 자막 추출 |
| [src/nodes.py](src/nodes.py) | 요약·브리핑 노드 |
| [src/storage.py](src/storage.py) | 날짜별 결과 저장/조회 |
| [src/run.py](src/run.py) | 터미널에서 브리핑 실행 |

## 준비

1. [uv](https://docs.astral.sh/uv/) 설치 후 `uv sync`
2. 프로젝트 루트에 `.env` 작성

```
OPENAI_API_KEY=...

# 선택: 유튜브가 IP를 차단했을 때 자막을 대신 받아옴
APIFY_API_TOKEN=...

# 선택: LangSmith 실행 기록
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://apac.api.smith.langchain.com
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...
```

`.env`에는 한글을 쓰지 마세요. Windows에서 `langgraph dev`가 파일을 읽지 못합니다.

## 사용법 (Windows)

**1. 브리핑 만들기** — 날마다 한 번

- `digest.bat` 더블클릭 (오늘 날짜, KST)
- 특정 날짜: `digest.bat 2026-09-29`

**2. 채팅하기**

1. `studio.bat` 더블클릭 → 브라우저에 LangGraph Studio가 열림 (검은 창은 닫지 말 것)
2. LangSmith 로그인 → 왼쪽 위에서 `chat` 선택 → Chat 모드
3. "오늘 시장 요약해줘", "삼성전자 얘기 뭐 나왔어?"처럼 질문

날짜를 말하지 않으면 가장 최근에 만든 브리핑을 기준으로 답합니다. 브리핑을 만든 날짜만 대화할 수 있습니다.

배치 파일 없이 직접 실행할 때 (PowerShell):

```powershell
$env:PYTHONUTF8=1
uv run python -m src.run 2026-09-29   # 브리핑
uv run langgraph dev                  # Studio
```

## 자막을 가져오는 순서와 비용

1. `.cache/transcripts/` 캐시
2. `youtube-transcript-api`로 직접 요청 (무료)
3. 유튜브가 IP를 차단하면 Apify Actor(`pintostudio/youtube-transcript-scraper`)로 대체
   - 영상당 약 $0.01, 무료 플랜 월 $5 크레딧 안에서 사용
   - 이번 달 사용액이 `APIFY_MONTHLY_BUDGET_USD`($4)를 넘으면 호출하지 않음

OpenAI 비용은 브리핑 실행(영상별 요약 + 브리핑 1회)과 채팅 질문마다 발생합니다.

## 알려진 제약

- 업로드 시각은 "3시간 전" 같은 표시로 계산한 근사값이라 1시간 정도 오차가 있습니다.
- 채널마다 최근 30개 영상만 보므로, 영상을 많이 올리는 채널(매일경제TV)은 지난 날짜 브리핑에서 빠질 수 있습니다.
- 영상별 요약에는 자막 앞 2만 자만 사용합니다 (`MAX_TRANSCRIPT_CHARS`). 채팅은 원문 전체를 검색합니다.
- 채팅 답변의 출처 링크가 가끔 다른 채널로 섞일 수 있습니다.

## 나중에 붙일 기능

- 브리핑 메시지 전송
- 매일 자동 실행
