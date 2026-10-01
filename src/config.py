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
MAX_VIDEOS_TOTAL = 5

# 마감 방송 우선 선별
# - 한국장은 당일 15:30(KST)에 끝나므로 그날 오후·저녁의 '마감' 영상이 당일 한국장을 다룬다.
# - 미국장은 다음 날 새벽 5~6시(KST)에 끝나므로 그날 오전의 미국 증시 영상은 미국 현지 '전날' 장을 다룬다.
#   미국장 마감 방송은 제목에 '마감'이 없는 경우가 많아, 오전 업로드 + 미국 증시 키워드로도 판별한다.
CLOSING_KEYWORDS = ["마감"]
US_MARKET_KEYWORDS = ["뉴욕증시", "뉴욕 증시", "미 증시", "미증시", "미국 증시", "미국증시", "월가", "나스닥", "다우", "S&P"]
US_CLOSE_UPLOAD_BEFORE_HOUR = 12  # 이 시각(KST) 전에 올라온 미국 증시 영상은 미국장 마감 방송으로 본다

# 자막 길이 제한 (문자 수): 너무 짧으면 건너뛰고, 너무 길면 잘라서 요약
MIN_TRANSCRIPT_CHARS = 300
MAX_TRANSCRIPT_CHARS = 20_000

# 받은 자막을 저장해 두는 폴더 (재실행 시 유튜브 재요청 방지)
TRANSCRIPT_CACHE_DIR = ".cache/transcripts"

# 날짜별 요약 결과(영상 목록·요약·브리핑)를 저장하는 폴더. 채팅 그래프가 여기서 읽는다.
DATA_DIR = "data"

# 숫자 검증 실패 등 실행 로그를 남기는 폴더
LOG_DIR = "logs"

# 유튜브가 IP를 차단했을 때 자막을 대신 가져올 Apify Actor (APIFY_API_TOKEN 필요)
APIFY_TRANSCRIPT_ACTOR = "pintostudio~youtube-transcript-scraper"

# Apify 월 예산 (USD). 무료 플랜 한도($5) 안에서만 쓰도록, 이번 달 사용액이 이 값을 넘으면 Apify 호출을 멈춘다.
APIFY_MONTHLY_BUDGET_USD = 4.0

# 모델 설정
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5.4-mini")
DIGEST_MODEL = os.getenv("DIGEST_MODEL", "gpt-5.4")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4")
