import yaml
from langchain_core.prompts import loading
from langchain_core.prompts.base import BasePromptTemplate
import os


def load_prompt(file_path, encoding="utf8") -> BasePromptTemplate:
    with open(file_path, "r", encoding=encoding) as f:
        config = yaml.safe_load(f)

    return loading.load_prompt_from_config(config)


def langsmith(project_name=None, set_enable=True):
    if set_enable:
        result = os.environ.get("LANGCHAIN_API_KEY")
        if result is None or result.strip() == "":
            print("LangChain API Key가 설정되지 않았습니다.")
            return
        os.environ["LANGCHAIN_ENDPOINT"] = (
            "https://api.smith.langchain.com"  # LangSmith API 엔드포인트
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "true"  # true: 활성화
        os.environ["LANGCHAIN_PROJECT"] = project_name  # 프로젝트명
        print(f"LangSmith 추적을 시작합니다.\n[프로젝트명]\n{project_name}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"  # false: 비활성화
        print("LangSmith 추적을 하지 않습니다.")
