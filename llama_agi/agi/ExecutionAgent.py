from langchain.agents import AgentExecutor, ZeroShotAgent, load_tools
from langchain.chains import LLMChain
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Optional, Union

from agi.task_prompts import LC_EXECUTION_PROMPT, LC_PREFIX, LC_SUFFIX
from agi.tools.NoteTakingTools import record_note, search_notes
from agi.tools.WebpageSearchTool import search_webpage


class BaseExecutionAgent:
    def __init__(
        self,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        model_name: str = "text-davinci-003",
    ) -> None:
        if llm:
            self._llm = llm
        elif model_name == "text-davinci-003":
            self._llm = OpenAI(temperature=0, model_name=model_name, max_tokens=512)
        else:
            self._llm = ChatOpenAI(temperature=0, model_name=model_name, max_tokens=512)

    def execute_task(
        self, objective: str, task: str, completed_tasks_summary: str
    ) -> str:
        raise NotImplementedError("execute_task not implemented in BaseExecutionAgent")


class SimpleExecutionAgent(BaseExecutionAgent):
    def __init__(
        self,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        model_name: str = "text-davinci-003",
    ) -> None:
        super().__init__(llm=llm, model_name=model_name)
        self._prompt_template = PromptTemplate(
            template=LC_EXECUTION_PROMPT,
            input_variables=["task", "objective", "completed_tasks_summary"],
        )
        self._execution_chain = LLMChain(llm=self._llm, prompt=self._prompt_template)

    def execute_task(
        self, objective: str, task: str, completed_tasks_summary: str
    ) -> str:
        result = self._execution_chain.predict(
            objective=objective,
            task=task,
            completed_tasks_summary=completed_tasks_summary,
        )
        return result


class ToolExecutionAgent(BaseExecutionAgent):
    def __init__(
        self,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        model_name: str = "text-davinci-003",
    ) -> None:
        super().__init__(llm=llm, model_name=model_name)
        # use some default langchain tools
        self._tools = load_tools(["google-search-results-json"])

        # add our custom tools
        self._tools = self._tools + [search_notes, record_note, search_webpage]

        # create the agent
        self._agent_prompt = ZeroShotAgent.create_prompt(
            self._tools,
            prefix=LC_PREFIX,
            suffix=LC_SUFFIX,
            input_variables=[
                "objective",
                "task",
                "agent_scratchpad",
                "completed_tasks_summary",
            ],
        )
        self._llm_chain = LLMChain(llm=self._llm, prompt=self._agent_prompt)
        self._agent = ZeroShotAgent(
            llm_chain=self._llm_chain, tools=self._tools, verbose=True
        )
        self._execution_chain = AgentExecutor.from_agent_and_tools(
            agent=self._agent, tools=self._tools, verbose=True
        )

    def execute_task(
        self, objective: str, task: str, completed_tasks_summary: str
    ) -> str:
        result = self._execution_chain.run(
            objective=objective,
            task=task,
            completed_tasks_summary=completed_tasks_summary,
        )
        return result
