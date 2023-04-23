from typing import Any, List, Optional, Union
from string import Formatter

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.chat_models.base import BaseChatModel

from llama_agi.default_task_prompts import LC_PREFIX, LC_SUFFIX
from llama_agi.execution_agent.base import BaseExecutionAgent, LlamaAgentPrompts


class ToolExecutionAgent(BaseExecutionAgent):
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
        self.agent_prefix = self.prompts.get("agent_prefix", LC_PREFIX)
        self.agent_suffix = self.prompts.get("agent_suffix", LC_SUFFIX)

        # create the agent
        input_variables = [
            fn for _, fn, _, _ in Formatter().parse(self.agent_prefix) if fn is not None
        ] + [
            fn for _, fn, _, _ in Formatter().parse(self.agent_suffix) if fn is not None
        ]
        self._agent_prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix=LC_PREFIX,
            suffix=LC_SUFFIX,
            input_variables=input_variables,
        )
        self._llm_chain = LLMChain(llm=self._llm, prompt=self._agent_prompt)
        self._agent = ZeroShotAgent(
            llm_chain=self._llm_chain, tools=self.tools, verbose=True
        )
        self._execution_chain = AgentExecutor.from_agent_and_tools(
            agent=self._agent, tools=self.tools, verbose=True
        )

    def execute_task(self, **prompt_kwargs: Any) -> str:
        """Execute a task, using tools."""
        result = self._execution_chain.run(**prompt_kwargs)
        return result
