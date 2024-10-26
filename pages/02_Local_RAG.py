import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from retriever import create_retriever
from common import load_prompt, langsmith
from dotenv import load_dotenv
import os

# API KEY 로드
load_dotenv()

# langsmith 설정
langsmith(project_name="[Project] PDF-RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("Local Model-based RAG")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "xionic", "ollama"], index=0)


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일 캐시 저장(시간이 오래 걸리는 작업 처리 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return create_retriever(file_path)


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# 체인 생성
def create_chain(retriever, model_name="gpt-4o"):
    # 프롬프트 생성(Create Prompt)
    if model_name == "xionic":
        # 프롬프트 생성
        prompt = load_prompt("./prompts/pdf-rag-xionic.yaml", encoding="utf-8")
        # 언어모델(LLM) 생성
        llm = ChatOpenAI(
            model_name="llama-3.1-xionic-ko-70b",
            base_url="http://sionic.tech:28000/v1",
            api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
        )
        retriever = retriever
    elif model_name == "ollama":
        # 프롬프트 생성
        prompt = load_prompt("./prompts/pdf-rag.yaml", encoding="utf-8")
        # 언어모델(LLM) 생성
        llm = ChatOllama(model="EEVE_Korean-10.8B:latest", temperature=0)
        retriever = retriever | format_doc
    else:
        # 프롬프트 생성
        prompt = load_prompt("./prompts/pdf-rag-ollama.yaml", encoding="utf-8")
        # 언어모델(LLM) 생성
        llm = ChatOpenAI(model_name=model_name, temperature=0)
        retriever = retriever

    # 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸림)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain


# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메세지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 사용자의 입력이 들어올 경우
if user_input:
    # 체인을 생성
    chain = st.session_state["chain"]
    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
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
    else:
        warning_msg.error("파일을 업로드 해주세요.")
