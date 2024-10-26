import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import glob
from common import load_prompt
from dotenv import load_dotenv
import os

# API KEY 로드
load_dotenv()

st.title("Custom Chat-GPT")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")

    prompt_files = glob.glob("./prompts/*.yaml")
    file_names = [os.path.splitext(os.path.basename(file))[0] for file in prompt_files]
    selected_prompt = st.selectbox("프롬프트를 선택해주세요", file_names, index=0)
    task_input = st.text_input("TASK 입력", "")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인 생성
def create_chain(prompt_file_path, task=""):
    # 프롬프트 적용
    prompt = load_prompt("./prompts/" + prompt_file_path + ".yaml", encoding="utf-8")
    if task:
        prompt = prompt.partial(task=task)

    # 모델(GPT)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser

    return chain


# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 사용자의 입력이 들어올 경우
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # 체인을 생성
    chain = create_chain(selected_prompt, task=task_input)
    # 스트리밍 호출
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화기록을 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
