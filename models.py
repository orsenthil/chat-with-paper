from enum import Enum

GPT3_LLM_MODEL = "gpt-3.5-turbo"
GPT4_LLM_MODEL = "gpt-4-turbo"
GPT4o_LLM_MODEL = "gpt-4o"


class LLMProviderEnum(str, Enum):
    GPT3 = "GPT-3"
    GPT4 = "GPT-4"
    GPT4o = "GPT-4o"
