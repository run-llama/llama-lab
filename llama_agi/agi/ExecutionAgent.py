from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from agi.task_prompts import LC_EXECUTION_PROMPT


class SimpleExecutionAgent:
    def __init__(self, llm=None, model_name="text-davinci-003"):
        if llm:
            self._llm = llm
        elif model_name == "text-davinci-003":
            self._llm = OpenAI(temperature=0, model_name=model_name, max_tokens=512)
        else:
            self._llm = ChatOpenAI(temperature=0, model_name=model_name, max_tokens=512 )

        self._prompt_template = PromptTemplate(
            template=LC_EXECUTION_PROMPT, 
            input_variables=["task", "objective", "completed_tasks_summary"]
        )
        self._execution_chain = LLMChain(llm=self._llm, prompt=self._prompt_template)
    
    def execute_task(self, objective, task, completed_tasks_summary):
        result = self._execution_chain.predict(
            objective=objective, 
            task=task, 
            completed_tasks_summary=completed_tasks_summary
        )
        return result
