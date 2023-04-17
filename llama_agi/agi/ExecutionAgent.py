from langchain.chains import LLMChain
from langchain.llms import OpenAI, BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Optional, Union

from agi.task_prompts import LC_EXECUTION_PROMPT


class SimpleExecutionAgent:
    def __init__(
        self,
        llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
        model_name: str = "text-davinci-003",
    ) -> None:
        if llm is not None:
            self._llm = llm
        elif model_name == "text-davinci-003":
            self._llm = OpenAI(temperature=0, model_name=model_name, max_tokens=512)
        else:
            self._llm = ChatOpenAI(temperature=0, model_name=model_name, max_tokens=512)

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
