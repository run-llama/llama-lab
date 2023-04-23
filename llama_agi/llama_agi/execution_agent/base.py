from abc import abstractmethod
from typing import Any, List, Optional, Union, TypedDict, NotRequired

from langchain.agents.tools import Tool
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI


class LlamaAgentPrompts(TypedDict):
    execution_prompt: NotRequired[str]
    agent_prefix: NotRequired[str]
    agent_suffix: NotRequired[str]


class BaseExecutionAgent:
    def __init__(
        self,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        model_name: str = "text-davinci-003",
        max_tokens: int = 512,
        prompts: Optional[LlamaAgentPrompts] = None,
        tools: Optional[List[Tool]] = None,
    ) -> None:
        if llm:
            self._llm = llm
        elif model_name == "text-davinci-003":
            self._llm = OpenAI(
                temperature=0, model_name=model_name, max_tokens=max_tokens
            )
        else:
            self._llm = ChatOpenAI(
                temperature=0, model_name=model_name, max_tokens=max_tokens
            )
        self.max_tokens = max_tokens
        self.prompts = prompts if prompts else {}
        self.tools = tools if tools else []

    @abstractmethod
    def execute_task(self, **prompt_kwargs: Any) -> str:
        """Execute a task."""
