# src/config.py
import os
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

# 요약 대상 금융·경제 뉴스 채널 (유튜브 핸들 → 표시 이름). 채널마다 그날 영상 1개씩 브리핑·채팅에 쓴다.
CHANNELS: dict[str, str] = {
    "@hkwowtv": "한국경제TV",
    "@3protv": "삼프로TV",
    "@koreanstockrider": "증시각도기TV",
}

# 채널당 최신순으로 훑어볼 최대 영상 수 (이 중 그날 올라온 영상이 선정 후보)
MAX_VIDEOS_PER_CHANNEL = 30

# 마감 방송 우선 선별
# - 한국장은 당일 15:30(KST)에 끝나므로 그날 오후·저녁의 '마감' 영상이 당일 한국장을 다룬다.
# - 미국장은 다음 날 새벽 5~6시(KST)에 끝나므로 그날 오전의 미국 증시 영상은 미국 현지 '전날' 장을 다룬다.
#   미국장 마감 방송은 제목에 '마감'이 없는 경우가 많아, 오전 업로드 + 미국 증시 키워드로도 판별한다.
CLOSING_KEYWORDS = ["마감", "클로징", "한국시황"]
US_MARKET_KEYWORDS = ["뉴욕증시", "뉴욕 증시", "미 증시", "미증시", "미국 증시", "미국증시", "월가", "나스닥", "다우", "S&P"]
US_CLOSE_UPLOAD_BEFORE_HOUR = 12  # 이 시각(KST) 전에 올라온 미국 증시 영상은 미국장 마감 방송으로 본다

# 영상별 요약 (문자 수 기준)
# - 이보다 짧은 자막은 요약하지 않는다
MIN_TRANSCRIPT_CHARS = 300
# - 이 길이 이하는 한 번에 요약하고, 넘으면 구간 수 = ceil(글자 수 / 이 값)으로 균등하게 나눠
#   구간별 요약 → 합치기 (예: 45,000자 → 15,000자 × 3구간). 구간 경계는 타임스탬프에 맞춘다.
#   긴 자막을 한 번에 넣으면 앞부분에 치우쳐 후반부가 빠지는 것을 실험으로 확인함 (experiments/compare_summary.py)
SINGLE_PASS_MAX_CHARS = 20_000

# 영상 자막(스크립트) 저장 폴더. 채팅 상담의 근거가 되는 핵심 데이터이며, 재실행 시 재요청도 막는다.
TRANSCRIPT_DIR = "data/transcripts"

# 토큰 비용을 고려해 이보다 긴 영상은 후보에서 뺀다 (초). 장 마감 생방송이 길어 2시간까지 허용
MAX_VIDEO_SECONDS = 2 * 60 * 60

# 날짜별 요약 결과(영상 목록·요약·브리핑)를 저장하는 폴더. 채팅 그래프가 여기서 읽는다.
DATA_DIR = "data"

# 자막 청크 검색용 벡터 DB와 로컬 임베딩 모델 (처음 한 번 약 2.2GB 내려받음)
CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "BAAI/bge-m3"

# 숫자 검증 실패 등 실행 로그를 남기는 폴더
LOG_DIR = "logs"

# 유튜브가 IP를 차단했을 때 자막을 대신 가져올 Apify Actor (APIFY_API_TOKEN 필요)
APIFY_TRANSCRIPT_ACTOR = "pintostudio~youtube-transcript-scraper"

# Apify 월 예산 (USD). 무료 플랜 한도($5) 안에서만 쓰도록, 이번 달 사용액이 이 값을 넘으면 Apify 호출을 멈춘다.
APIFY_MONTHLY_BUDGET_USD = 4.0

# 모델 설정
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5.4-mini")
SELECT_MODEL = os.getenv("SELECT_MODEL", "gpt-5.4-mini")  # 마감 방송 외 나머지 영상을 제목으로 고르는 모델
DIGEST_MODEL = os.getenv("DIGEST_MODEL", "gpt-5.4")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.4")
