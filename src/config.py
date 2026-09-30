# src/config.py
import os
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

# 요약 대상 금융·경제 뉴스 채널 (유튜브 핸들 → 표시 이름)
CHANNELS: dict[str, str] = {
    "@hkwowtv": "한국경제TV",
    "@MKeconomy_TV": "매일경제TV",
    "@3protv": "삼프로TV",
}

# 채널당 최신순으로 훑어볼 최대 영상 수
MAX_VIDEOS_PER_CHANNEL = 30

# 하루에 요약할 최대 영상 수 (자막 요청을 줄여 유튜브 IP 차단 위험을 낮춤)
MAX_VIDEOS_TOTAL = 10

# 자막 길이 제한 (문자 수): 너무 짧으면 건너뛰고, 너무 길면 잘라서 요약
MIN_TRANSCRIPT_CHARS = 300
MAX_TRANSCRIPT_CHARS = 20_000

# 받은 자막을 저장해 두는 폴더 (재실행 시 유튜브 재요청 방지)
TRANSCRIPT_CACHE_DIR = ".cache/transcripts"

# 유튜브가 IP를 차단했을 때 자막을 대신 가져올 Apify Actor (APIFY_API_TOKEN 필요)
APIFY_TRANSCRIPT_ACTOR = "pintostudio~youtube-transcript-scraper"

# Apify 월 예산 (USD). 무료 플랜 한도($5) 안에서만 쓰도록, 이번 달 사용액이 이 값을 넘으면 Apify 호출을 멈춘다.
APIFY_MONTHLY_BUDGET_USD = 4.0

# 모델 설정
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5.4-mini")
DIGEST_MODEL = os.getenv("DIGEST_MODEL", "gpt-5.4")
