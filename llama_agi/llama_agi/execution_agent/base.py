from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union, TypedDict, NotRequired

from langchain.agents.tools import Tool
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI


class LlamaAgentPrompts(TypedDict):
    execution_prompt: NotRequired[str]
    agent_prefix: NotRequired[str]
    agent_suffix: NotRequired[str]


class BaseExecutionAgent:
    """Base Execution Agent
    
    Args:
        llm (Union[BaseLLM, BaseChatModel]): The langchain LLM class to use.
        model_name: (str): The name of the OpenAI model to use, if the LLM is 
        not provided.
        max_tokens: (int): The maximum number of tokens the LLM can generate.
        prompts: (LlamaAgentPrompts): The prompt templates used during execution.
        tools: (List[Tool]): The list of langchain tools for the execution 
        agent to use.
    """
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
    def execute_task(self, **prompt_kwargs: Any) -> Dict[str, str]:
        """Execute a task."""
