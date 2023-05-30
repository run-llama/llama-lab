from typing import Any, Dict, List, Optional, Union
from string import Formatter

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.llms import BaseLLM
from langchain.chat_models.base import BaseChatModel

from llama_agi.execution_agent.base import BaseExecutionAgent, LlamaAgentPrompts


class ToolExecutionAgent(BaseExecutionAgent):
    """Tool Execution Agent

    This agent is a wrapper around the zero-shot agent from Langchain. Using
    a set of tools, the agent is expected to carry out and complete some task
    that will help achieve an overall objective.

    The agents overall behavior is controlled by the LlamaAgentPrompts.agent_prefix
    and LlamaAgentPrompts.agent_suffix prompt templates.

    The execution template kwargs are automatically extracted and expected to be
    specified in execute_task().

    execute_task() also returns the intermediate steps, for additional debugging and is
    used for the streamlit example.

    Args:
        llm (Union[BaseLLM, BaseChatModel]): The langchain LLM class to use.
        model_name: (str): The name of the OpenAI model to use, if the LLM is
        not provided.
        max_tokens: (int): The maximum number of tokens the LLM can generate.
        prompts: (LlamaAgentPrompts): The prompt templates used during execution.
        The Tool Execution Agent uses LlamaAgentPrompts.agent_prefix and
        LlamaAgentPrompts.agent_suffix.
        tools: (List[Tool]): The list of langchain tools for the execution agent to use.
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
        self.agent_prefix = self.prompts.agent_prefix
        self.agent_suffix = self.prompts.agent_suffix

        # create the agent
        input_variables = [
            fn for _, fn, _, _ in Formatter().parse(self.agent_prefix) if fn is not None
        ] + [
            fn for _, fn, _, _ in Formatter().parse(self.agent_suffix) if fn is not None
        ]
        self._agent_prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix=self.agent_prefix,
            suffix=self.agent_suffix,
            input_variables=input_variables,
        )
        self._llm_chain = LLMChain(llm=self._llm, prompt=self._agent_prompt)
        self._agent = ZeroShotAgent(
            llm_chain=self._llm_chain, tools=self.tools, verbose=True
        )
        self._execution_chain = AgentExecutor.from_agent_and_tools(
            agent=self._agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
        )

    def execute_task(self, **prompt_kwargs: Any) -> Dict[str, str]:
        """Execute a task, using tools."""
        result = self._execution_chain(prompt_kwargs)
        return result
