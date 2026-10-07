# src/facts.py
"""숫자 레이어: 브리핑의 시장 숫자(지수, 수급, 환율, 종목 등락률)는 LLM이 아니라 코드가 확정한다.

데이터 소스
- 지수·투자자별 수급·종목 시세: KRX (pykrx, .env의 KRX_ID / KRX_PW로 로그인)
- 원/달러: 한국은행 ECOS 731Y001 / 0000001 (원/미국달러 매매기준율, .env의 ECOS_API_KEY)
"""
import os
import re
from datetime import date, timedelta
from functools import lru_cache

import httpx

INDEX_CODES = {"코스피": "1001", "코스닥": "2001"}
MARKETS = {"코스피": "KOSPI", "코스닥": "KOSDAQ"}
INVESTOR_COLUMNS = {"외국인": "외국인합계", "기관": "기관합계", "개인": "개인"}

# 자막에서 종목을 찾을 때 쓰는 줄임말 → 정식 종목명
STOCK_ALIASES = {
    "삼전": "삼성전자",
    "하이닉스": "SK하이닉스",
    "하닉": "SK하이닉스",
    "엘지엔솔": "LG에너지솔루션",
    "엘엔솔": "LG에너지솔루션",
    "현차": "현대차",
}
# 일반 단어와 겹쳐 자막 매칭에서 오탐이 나는 2글자 종목명은 제외하고, 확실한 것만 허용
SHORT_NAME_ALLOWLIST = {"기아"}
MIN_STOCK_MENTIONS = 2
MAX_STOCKS = 12


def _krx():
    """pykrx는 import 시점에 KRX 로그인을 하므로 .env가 로드된 뒤 사용하는 시점에 불러온다."""
    from pykrx import stock
    return stock


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


# ---------- 개별 숫자 ----------

def index_levels(target: date) -> dict:
    """코스피·코스닥 시가/고가/저가/종가와 전일 대비 등락."""
    stock = _krx()
    result = {}
    for name, code in INDEX_CODES.items():
        df = stock.get_index_ohlcv(_ymd(target - timedelta(days=15)), _ymd(target), code)
        if df.empty or df.index[-1].date() != target:
            raise ValueError(f"{target} {name} 지수 데이터 없음 (휴장일일 수 있음)")
        today, prev = df.iloc[-1], df.iloc[-2]
        result[name] = {
            "open": round(float(today["시가"]), 2),
            "high": round(float(today["고가"]), 2),
            "low": round(float(today["저가"]), 2),
            "close": round(float(today["종가"]), 2),
            "prev_close": round(float(prev["종가"]), 2),
            "change": round(float(today["종가"] - prev["종가"]), 2),
            "change_pct": round(float((today["종가"] / prev["종가"] - 1) * 100), 2),
        }
    return result


def investor_flows(target: date) -> dict:
    """시장별 투자자 순매수 금액 (억 원, 음수 = 순매도)."""
    stock = _krx()
    result = {}
    for name, market in MARKETS.items():
        df = stock.get_market_trading_value_by_date(_ymd(target), _ymd(target), market)
        if df.empty:
            raise ValueError(f"{target} {name} 수급 데이터 없음")
        row = df.iloc[-1]
        result[name] = {who: round(float(row[col]) / 1e8) for who, col in INVESTOR_COLUMNS.items()}
    return result


def usdkrw(target: date) -> dict:
    """원/달러 매매기준율과 전일 대비 변화 (ECOS 731Y001 / 0000001)."""
    key = os.environ["ECOS_API_KEY"]
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/30/731Y001/D/"
        f"{_ymd(target - timedelta(days=15))}/{_ymd(target)}/0000001"
    )
    rows = httpx.get(url, timeout=30).json().get("StatisticSearch", {}).get("row", [])
    if len(rows) < 2 or rows[-1]["TIME"] != _ymd(target):
        raise ValueError(f"{target} 원/달러 매매기준율 데이터 없음")
    rate, prev = float(rows[-1]["DATA_VALUE"]), float(rows[-2]["DATA_VALUE"])
    return {"rate": rate, "prev": prev, "change": round(rate - prev, 2)}


@lru_cache(maxsize=4)
def ticker_names(target_ymd: str) -> dict[str, str]:
    """종목명 → 티커 (코스피 + 코스닥)."""
    stock = _krx()
    names = {}
    for market in MARKETS.values():
        for ticker in stock.get_market_ticker_list(target_ymd, market=market):
            names[stock.get_market_ticker_name(ticker)] = ticker
    return names


def stock_change(names: list[str], target: date) -> dict:
    """종목별 종가와 전일 대비 등락률. 찾지 못한 종목은 건너뛴다."""
    stock = _krx()
    tickers = ticker_names(_ymd(target))
    result = {}
    for name in names:
        ticker = tickers.get(name)
        if not ticker:
            continue
        df = stock.get_market_ohlcv(_ymd(target), _ymd(target), ticker)
        if df.empty:
            continue
        row = df.iloc[-1]
        result[name] = {"ticker": ticker, "close": int(row["종가"]), "change_pct": round(float(row["등락률"]), 2)}
    return result


# 종목명 바로 뒤에 올 수 있는 조사 첫 글자. 이 밖의 한글이 붙으면 다른 단어의 일부로 본다 ('아스트라' ≠ '아스트')
_PARTICLE_START = set("가이은는을를도와과의에로만까랑하보처부께요죠")


def _starts_word(text: str, start: int) -> bool:
    """앞 글자가 한글·영문·숫자가 아니면 단어의 시작이다 ('SK하이닉스' 속 '이닉스'는 단어 시작이 아님)."""
    prev = text[start - 1: start]
    return not prev or not (prev.isalnum() or "가" <= prev <= "힣")


def _count_as_word(text: str, name: str) -> int:
    count = 0
    for m in re.finditer(re.escape(name), text):
        nxt = text[m.end(): m.end() + 1]
        if not _starts_word(text, m.start()):
            continue
        if not nxt or not ("가" <= nxt <= "힣") or nxt in _PARTICLE_START:
            count += 1
    return count


def extract_stocks(texts: list[str], target: date, exclude: set[str] = frozenset()) -> list[str]:
    """자막에서 언급된 상장 종목명을 언급 횟수순으로 고른다.

    더 긴 종목명·줄임말의 일부로만 나온 경우('SK하이닉스' 속 '이닉스')는 세지 않고,
    exclude(방송 채널명처럼 상장사지만 종목 언급이 아닌 이름)는 제외한다.
    """
    tickers = ticker_names(_ymd(target))
    joined = "\n".join(texts)
    raw = {
        name: n for name in tickers
        if name not in exclude
        and (len(name) >= 3 or name in SHORT_NAME_ALLOWLIST)
        and name in joined
        and (n := _count_as_word(joined, name))
    }
    longer_words = set(raw) | set(STOCK_ALIASES)
    counts: dict[str, int] = {}
    for name, n in raw.items():
        inside_longer = sum(joined.count(w) for w in longer_words if w != name and name in w)
        if n - inside_longer > 0:
            counts[name] = n - inside_longer
    for alias, name in STOCK_ALIASES.items():
        if name not in tickers:
            continue
        n = joined.count(alias)
        if alias in name:
            n -= joined.count(name)  # 정식 이름('SK하이닉스') 속 줄임말('하이닉스')은 위에서 이미 셌다
        if n > 0:
            counts[name] = counts.get(name, 0) + n
    ranked = sorted((c, name) for name, c in counts.items() if c >= MIN_STOCK_MENTIONS)
    return [name for _, name in reversed(ranked)][:MAX_STOCKS]


# ---------- 묶음 ----------

def get_market_facts(target_date: str, stock_names: list[str]) -> dict:
    """그날의 확정 숫자를 모은다. 실패한 항목은 errors에 남기고 나머지는 계속 채운다."""
    target = date.fromisoformat(target_date)
    facts: dict = {"date": target_date, "indices": {}, "flows": {}, "usdkrw": None, "stocks": {}, "errors": []}
    for key, fn in (
        ("indices", lambda: index_levels(target)),
        ("flows", lambda: investor_flows(target)),
        ("usdkrw", lambda: usdkrw(target)),
        ("stocks", lambda: stock_change(stock_names, target)),
    ):
        try:
            facts[key] = fn()
        except Exception as e:
            facts["errors"].append(f"{key}: {type(e).__name__}: {e}")
    return facts


# ---------- 표시 ----------

def _signed(v: float, digits: int = 2) -> str:
    return f"{v:+,.{digits}f}"


def _eok(v: int) -> str:
    """억 원 단위 정수를 '2조 646억' 형식으로."""
    sign = "-" if v < 0 else "+"
    jo, eok = divmod(abs(v), 10000)
    return f"{sign}{jo}조 {eok:,}억" if jo else f"{sign}{eok:,}억"


def render_numbers(facts: dict) -> str:
    """브리핑 맨 위 '오늘의 숫자' 섹션 (코드가 생성, LLM 관여 없음)."""
    lines = [f"## 오늘의 숫자 ({facts['date']} 한국장 마감 기준)", ""]

    if facts["indices"]:
        lines += ["| 지수 | 종가 | 전일 대비 | 장중 고가 / 저가 |", "|---|---|---|---|"]
        for name, x in facts["indices"].items():
            lines.append(
                f"| {name} | {x['close']:,.2f} | {_signed(x['change'])} ({_signed(x['change_pct'])}%) "
                f"| {x['high']:,.2f} / {x['low']:,.2f} |"
            )
        lines.append("")

    if facts["flows"]:
        lines += ["| 순매수 (원) | 외국인 | 기관 | 개인 |", "|---|---|---|---|"]
        for name, x in facts["flows"].items():
            lines.append(f"| {name} | {_eok(x['외국인'])} | {_eok(x['기관'])} | {_eok(x['개인'])} |")
        lines.append("")

    if facts["usdkrw"]:
        x = facts["usdkrw"]
        lines += [f"- **원/달러 매매기준율**: {x['rate']:,.1f}원 (전일 대비 {_signed(x['change'], 1)}원)", ""]

    if facts["stocks"]:
        lines += ["| 종목 | 종가 | 등락률 |", "|---|---|---|"]
        for name, x in facts["stocks"].items():
            lines.append(f"| {name} | {x['close']:,}원 | {_signed(x['change_pct'])}% |")
        lines.append("")

    lines.append("*출처: KRX(지수·수급·종목), 한국은행 ECOS(환율). 이 섹션은 코드가 생성합니다.*")
    if facts["errors"]:
        lines.append(f"*수집 실패: {'; '.join(facts['errors'])}*")
    return "\n".join(lines)


def facts_for_prompt(facts: dict) -> str:
    """LLM에 넘길 확정 숫자 목록."""
    return render_numbers(facts)


FACT_RULES = """[숫자 규칙 — 반드시 지킬 것]
- 아래 '확정 숫자'에 있는 항목(코스피·코스닥 지수와 등락률, 투자자별 순매수, 원/달러, 표에 있는 종목의 등락률)은
  방송에서 다른 숫자를 말했더라도 확정 숫자만 사용합니다. 방송 수치를 이 항목들에 쓰지 않습니다.
- 지수 수준은 종가·시가·장중 고가·저가만 씁니다. '장중 6,780선까지' 같은 방송 속 중간 수치는 쓰지 않습니다.
- 수급 금액은 '2조 646억 원'처럼 확정 숫자 그대로 쓰고, 순매수/순매도 방향을 바꾸지 않습니다.
- 확정 숫자에 없는 숫자(실적, 목표주가, 해외 지표, 금리 등)는 방송 요약에 있는 그대로 쓰되 출처를 답니다.
- 브리핑 본문에는 '오늘의 숫자' 섹션을 쓰지 않습니다. 코드가 따로 붙입니다."""


# ---------- 검증 ----------

_PCT = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
# 뒤에 억·조·원·만이 붙은 숫자는 금액이라 지수 수준으로 보지 않는다 ('-1,305억 원')
_INDEX_LEVEL = re.compile(r"(?<![\d,.])(\d{1,2},?\d{3}(?:\.\d+)?)(?![\d,]|\s*(?:억|조|원|만|천))\s*(선|포인트|pt)?")
# 이런 말이 있는 문장의 수급은 여러 날 누적이라 그날 순매수와 비교하지 않는다
_PERIOD_WORDS = ("이후", "누적", "올해", "연초", "연간", "지난", "들어", "동안", "개월", "주간", "월간", "이번 달", "이번 주")
# 부호가 없으면 이런 말이 뒤따를 때만 등락률로 본다 ('가동률 98%'는 등락률이 아님)
_CHANGE_WORDS = ("상승", "하락", "올라", "올랐", "오른", "내려", "내렸", "내린", "떨어", "급등", "급락", "강세", "약세", "반등", "밀려", "밀렸")
_FX = re.compile(r"(1,?\d{3}(?:\.\d+)?)\s*원")
_AMOUNT = re.compile(r"(?:(\d+(?:\.\d+)?)\s*조)?\s*(?:(\d+)\s*천)?\s*(?:(\d[\d,]*)\s*)?억")
# 문장 끝 마침표에서만 나눈다 ('1.4%' 같은 소수점에서 자르지 않도록)
_SENTENCE_END = re.compile(r"(?<=[다요음함됨])\.(?=\s|$)|[\n!?]")
# 이런 말이 있는 문장의 지수 수준은 전망·지지선이라 종가와 비교하지 않는다
_OUTLOOK_WORDS = ("전망", "재도전", "목표", "가능", "예상", "견해", "지지", "저항", "회복", "돌파 시")


def _window(sentence: str, start: int, stop_words: list[str], size: int = 45) -> str:
    """키워드 뒤 size자, 단 다른 대상 키워드가 나오면 그 앞까지."""
    seg = sentence[start: start + size]
    cut = min((i for w in stop_words if (i := seg.find(w, 1)) > 0), default=len(seg))
    return seg[:cut]


def _is_change(p: re.Match, win: str) -> bool:
    """이 %가 등락률인지: 부호가 있거나, 바로 뒤에 상승·하락 같은 말이 온다."""
    return p.group(1)[0] in "+-" or any(w in win[p.end(): p.end() + 8] for w in _CHANGE_WORDS)


def _around(sentence: str, m: re.Match) -> str:
    """오류 메시지에 보여줄 키워드 주변 문맥."""
    return sentence[max(0, m.start() - 10): m.end() + 50].strip()


def _parse_eok(m: re.Match) -> float | None:
    jo, cheon, eok = m.group(1), m.group(2), m.group(3)
    if not (jo or cheon or eok):
        return None
    total = float(jo or 0) * 10000 + int(cheon or 0) * 1000
    if eok:
        total += int(eok.replace(",", ""))
    return total


def verify(text: str, facts: dict) -> list[str]:
    """LLM이 쓴 브리핑 본문의 숫자가 확정 숫자와 맞는지 검사해 오류 목록을 돌려준다."""
    errors: list[str] = []
    indices, flows, fx, stocks = facts["indices"], facts["flows"], facts["usdkrw"], facts["stocks"]
    entity_words = list(indices) + list(stocks) + list(INVESTOR_COLUMNS) + ["원/달러", "환율"]

    for sentence in _SENTENCE_END.split(text):
        if not sentence or not sentence.strip():
            continue
        is_outlook = any(w in sentence for w in _OUTLOOK_WORDS)
        is_period = any(w in sentence for w in _PERIOD_WORDS)
        # 지수: 등락률과 수준. '지난주 코스피 1% 하락' 같은 기간 등락률은 그날 숫자와 비교하지 않는다
        for name, x in indices.items():
            if is_period:
                continue
            for m in re.finditer(name, sentence):
                win = _window(sentence, m.end(), [w for w in entity_words if w != name])
                for p in _PCT.finditer(win):
                    if abs(abs(float(p.group(1))) - abs(x["change_pct"])) > 0.011:
                        errors.append(f"{name} 등락률 {p.group(0)} ≠ 확정 {x['change_pct']:+.2f}% — \"{_around(sentence, m)}\"")
                if is_outlook:
                    continue
                for lv in _INDEX_LEVEL.finditer(win):
                    v = float(lv.group(1).replace(",", ""))
                    tol = 10 if lv.group(2) == "선" else 1
                    if v < 1000 or any(abs(v - x[k]) <= tol or (tol == 10 and 0 <= x[k] - v < 10)
                                       for k in ("open", "high", "low", "close")):
                        continue
                    errors.append(f"{name} 지수 {lv.group(0).strip()} — 확정 시가/고가/저가/종가와 불일치 \"{_around(sentence, m)}\"")

        # 종목 등락률: 종목명 바로 뒤의 첫 번째 % 하나만 (그 뒤는 다른 종목 숫자일 수 있음)
        for name, x in stocks.items():
            for m in re.finditer(re.escape(name), sentence):
                if not _starts_word(sentence, m.start()):
                    continue  # 더 긴 종목명의 일부 ('SK하이닉스' 속 '이닉스')
                win = _window(sentence, m.end(), [w for w in entity_words if w != name], size=30)
                p = _PCT.search(win)
                if p and not _is_change(p, win):
                    continue
                if p and abs(abs(float(p.group(1))) - abs(x["change_pct"])) > 0.051:
                    errors.append(f"{name} 등락률 {p.group(0)} ≠ 확정 {x['change_pct']:+.2f}% — \"{_around(sentence, m)}\"")

        # 환율
        if fx:
            for word in ("원/달러", "환율"):
                for m in re.finditer(word, sentence):
                    for f in _FX.finditer(_window(sentence, m.end(), [w for w in entity_words if w != word])):
                        v = float(f.group(1).replace(",", ""))
                        if abs(v - fx["rate"]) > 0.11:
                            errors.append(f"원/달러 {f.group(0)} ≠ 확정 {fx['rate']:,.1f}원 — \"{_around(sentence, m)}\"")

        # 투자자별 순매수: 직전에 언급된 시장 기준 (없으면 코스피). 기간 누적 수급 문장은 제외
        if any(w in sentence for w in _PERIOD_WORDS):
            continue
        for who in INVESTOR_COLUMNS:
            for m in re.finditer(who, sentence):
                before = sentence[: m.start()]
                market = max(flows, key=lambda k: before.rfind(k)) if flows else None
                if not market:
                    continue
                if before.rfind("코스피") < 0 and before.rfind("코스닥") < 0:
                    market = "코스피" if "코스피" in flows else market
                fact = flows[market][who]
                win = _window(sentence, m.end(), [w for w in INVESTOR_COLUMNS if w != who] + list(indices))
                for a in _AMOUNT.finditer(win):
                    v = _parse_eok(a)
                    # 선물 수급은 확정 숫자(현물)와 다른 항목이라 비교하지 않는다
                    if v is None or v < 10 or "선물" in win[: a.start()]:
                        continue
                    if abs(v - abs(fact)) > max(abs(fact) * 0.05, 10):
                        errors.append(f"{market} {who} 금액 {a.group(0).strip()} ≠ 확정 {_eok(fact)} 원 — \"{_around(sentence, m)}\"")
                if fact < 0 and "순매수" in win and "매도" not in win:
                    errors.append(f"{market} {who}는 순매도인데 순매수로 표기 — \"{_around(sentence, m)}\"")
                if fact > 0 and "순매도" in win and "매수" not in win:
                    errors.append(f"{market} {who}는 순매수인데 순매도로 표기 — \"{_around(sentence, m)}\"")

    return list(dict.fromkeys(errors))
