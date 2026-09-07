import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# 1. 사용자 입력(messages에 들어있음) 에 맞는 주제로, 농담을 생성해 줌.
from langchain_core.messages import AIMessage
from openai import OpenAI
from src.state import JokeState

client = OpenAI()
MODEL_NAME = "gpt-4o-mini"

def joke_node(state: JokeState) -> dict:
    """사용자 메시지를 기반으로 OpenAI API를 호출해 농담을 생성하는 노드"""
    messages = state["messages"]
    
    # 가장 마지막 사용자 메시지 가져오기
    latest_user_input = messages[-1].content if messages else "재치 있는 농담"

    prompt = f"""
    사용자의 다음 요청이나 상황을 바탕으로, 그에 어울리는 센스 있고 재치 있는 농담을 하나 생성해 주세요.
    
    요청/상황: {latest_user_input}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "당신은 사람들을 유쾌하게 만들어주는 재치 있는 농담 전문 에이전트입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )
    
    joke_text = response.choices[0].message.content.strip()
    
    # 상태 업데이트: 생성된 농담 문자열과 대화 메시지 기록을 함께 반환
    return {
        "joke": joke_text,
        "messages": [AIMessage(content=f"[생성된 농담]\n{joke_text}")]
    }


# 2. 1번 노드가 생성한 농담을 평가해 줌(feedback, score 를 포함해야함.)
def eval_node(state: JokeState):
    #  messages에 농담, 평가, 점수까지 모두 담아서 보냄
    pass