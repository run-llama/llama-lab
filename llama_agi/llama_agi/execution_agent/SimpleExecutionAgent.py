from typing import Any, Dict, List, Optional, Union
from string import Formatter

from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate

from llama_agi.execution_agent.base import BaseExecutionAgent, LlamaAgentPrompts


class SimpleExecutionAgent(BaseExecutionAgent):
    """Simple Execution Agent

    This agent uses an LLM to execute a basic action without tools.
    The LlamaAgentPrompts.execution_prompt defines how this execution agent
    behaves.

    Usually, this is used for simple tasks, like generating the initial list of tasks.

    The execution template kwargs are automatically extracted and expected to be
    specified in execute_task().

    Args:
        llm (Union[BaseLLM, BaseChatModel]): The langchain LLM class to use.
        model_name: (str): The name of the OpenAI model to use, if the LLM is
        not provided.
        max_tokens: (int): The maximum number of tokens the LLM can generate.
        prompts: (LlamaAgentPrompts): The prompt templates used during execution.
        The only prompt used byt the SimpleExecutionAgent is
        LlamaAgentPrompts.execution_prompt.
    """

    def __init__(
        self,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        model_name: str = "text-davinci-003",
        max_tokens: int = 512,
        prompts: LlamaAgentPrompts = LlamaAgentPrompts(),
        tools: Optional[List[Tool]] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            model_name=model_name,
            max_tokens=max_tokens,
            prompts=prompts,
            tools=tools,
        )

        self.execution_prompt = self.prompts.execution_prompt
        input_variables = [
            fn
            for _, fn, _, _ in Formatter().parse(self.execution_prompt)
            if fn is not None
        ]
        self._prompt_template = PromptTemplate(
            template=self.execution_prompt,
            input_variables=input_variables,
        )
        self._execution_chain = LLMChain(llm=self._llm, prompt=self._prompt_template)

    def execute_task(self, **prompt_kwargs: Any) -> Dict[str, str]:
        """Execute a task."""
        result = self._execution_chain.predict(**prompt_kwargs)
        return {"output": result}
