from typing import Any, List, Optional, Union
from string import Formatter

from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import PromptTemplate

from llama_agi.execution_agent.base import BaseExecutionAgent, LlamaAgentPrompts
from llama_agi.default_task_prompts import LC_EXECUTION_PROMPT


class SimpleExecutionAgent(BaseExecutionAgent):
    def __init__(
        self,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        model_name: str = "text-davinci-003",
        max_tokens: int = 512,
        prompts: Optional[LlamaAgentPrompts] = None,
        tools: Optional[List[Tool]] = None,
    ) -> None:
        super().__init__(
            llm=llm,
            model_name=model_name,
            max_tokens=max_tokens,
            prompts=prompts,
            tools=tools,
        )

        self.execution_prompt = self.prompts.get(
            "execution_prompt", LC_EXECUTION_PROMPT
        )

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

    def execute_task(self, **prompt_kwargs: Any) -> str:
        """Execute a task."""
        result = self._execution_chain.predict(**prompt_kwargs)
        return result
