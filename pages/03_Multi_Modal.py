import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from multi_modal import MultiModal
from common import langsmith
from dotenv import load_dotenv
import os

# API KEY 로드
load_dotenv()

# langsmith 설정
langsmith(project_name="[Project] 이미지 인식")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("Image recognition-Based ChatBot")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성
    st.session_state["messages"] = []

# 탭 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    # 시스템 프롬프트 추가
    system_prompt = st.text_area(
        "시스템 프롬프트",
        "당신은 상품 정보를 이미지를 기반으로 분석하는 AI입니다.\n주어진 이미지를 바탕으로 분석한 상품 정보를 알려주세요.\n인스타그램에 마케팅할 수 있게 마케팅 문구를 작성해주세요.\n마케팅 문구 작성 후 마지막 문장으로 해시태그도 넣어주세요.",
        height=400,
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        main_tab2.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이미지 캐시 저장(시간이 오래 걸리는 작업 처리 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    # 업로드한 이미지를 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# 체인 생성
def generate_answer(image_filepath, system_prompt, user_prompt, model_name="gpt-4o"):
    # 프롬프트 생성(Create Prompt)
    llm = ChatOpenAI(
        temperature=0,
        model_name=model_name,
    )
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)
    answer = multimodal.stream(image_filepath)
    return answer


# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메세지를 띄우기 위한 빈 영역
warning_msg = main_tab2.empty()

# 이미지가 업로드 되는 경우
if uploaded_file:
    image_filepath = process_imagefile(uploaded_file)
    main_tab1.image(image_filepath)

# 사용자의 입력이 들어올 경우
if user_input:
    if uploaded_file:
        # 답변 요청
        response = generate_answer(
            image_filepath, system_prompt, user_input, selected_model
        )
        # 사용자의 입력
        main_tab2.chat_message("user").write(user_input)

        with main_tab2.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록을 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        warning_msg.error("이미지를 업로드 해주세요.")
