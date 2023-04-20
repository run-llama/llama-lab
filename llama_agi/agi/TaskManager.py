import re
import json
from typing import List, Tuple
from llama_index import Document
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt

from agi.utils import initialize_task_list_index
from agi.task_prompts import (
    DEFAULT_TASK_PRIORITIZE_TMPL,
    DEFAULT_REFINE_TASK_PRIORITIZE_TMPL,
    DEFAULT_TASK_CREATE_TMPL,
    DEFAULT_REFINE_TASK_CREATE_TMPL,
    NO_COMPLETED_TASKS_SUMMARY,
)


class TaskManager:
    def __init__(self, tasks: List[str]) -> None:
        self.current_tasks = [Document(x) for x in tasks]
        self.completed_tasks: List[Document] = []
        self.current_tasks_index = initialize_task_list_index(
            self.current_tasks, index_path="current_tasks_index.json"
        )
        self.completed_tasks_index = initialize_task_list_index(
            self.completed_tasks, index_path="completed_tasks_index.json"
        )

    def _get_task_create_templates(
        self, prev_task: str, prev_result: str
    ) -> Tuple[QuestionAnswerPrompt, RefinePrompt]:
        text_qa_template = DEFAULT_TASK_CREATE_TMPL.format(
            prev_result=prev_result,
            prev_task=prev_task,
            query_str="{query_str}",
            context_str="{context_str}",
        )
        llama_text_qa_template = QuestionAnswerPrompt(text_qa_template)

        refine_template = DEFAULT_REFINE_TASK_CREATE_TMPL.format(
            prev_result=prev_result,
            prev_task=prev_task,
            query_str="{query_str}",
            context_msg="{context_msg}",
            existing_answer="{existing_answer}",
        )
        llama_refine_template = RefinePrompt(refine_template)

        return (llama_text_qa_template, llama_refine_template)

    def _get_task_prioritize_templates(
        self,
    ) -> Tuple[QuestionAnswerPrompt, RefinePrompt]:
        return (
            QuestionAnswerPrompt(DEFAULT_TASK_PRIORITIZE_TMPL),
            RefinePrompt(DEFAULT_REFINE_TASK_PRIORITIZE_TMPL),
        )

    def parse_task_list(self, task_list_str: str) -> List[str]:
        # Try to parse lists with json, fallback to regex
        new_tasks: List[str] = []
        try:
            new_tasks = json.loads(str(new_tasks))
            new_tasks = [x.strip() for x in new_tasks if len(x.strip()) > 10]
        except Exception:
            new_tasks = str(new_tasks).split("\n")
            new_tasks = [
                re.sub(r"^[0-9]+\.", "", x).strip()
                for x in str(new_tasks)
                if len(x.strip()) > 10 and x[0].isnumeric()
            ]
        return new_tasks

    def get_completed_tasks_summary(self) -> str:
        if len(self.completed_tasks) == 0:
            return NO_COMPLETED_TASKS_SUMMARY
        summary = self.completed_tasks_index.query(
            "Summarize the current completed tasks", response_mode="tree_summarize"
        )
        return str(summary)

    def prioritize_tasks(self, objective: str) -> None:
        (text_qa_template, refine_template) = self._get_task_prioritize_templates()
        prioritized_tasks = self.current_tasks_index.query(
            objective,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
        )

        new_tasks = []
        for task in str(prioritized_tasks).split("\n"):
            task = re.sub(r"^[0-9]+\.", "", task).strip()
            if len(task) > 10:
                new_tasks.append(task)
        self.current_tasks = [Document(x) for x in new_tasks]
        self.current_tasks_index = initialize_task_list_index(self.current_tasks)

    def generate_new_tasks(
        self, objective: str, prev_task: str, prev_result: str
    ) -> None:
        (text_qa_template, refine_template) = self._get_task_create_templates(
            prev_task, prev_result
        )
        task_list_response = self.completed_tasks_index.query(
            objective,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
        )
        new_tasks = self.parse_task_list(str(task_list_response))
        self.add_new_tasks(new_tasks)

    def get_next_task(self) -> str:
        next_task = self.current_tasks.pop().get_text()
        self.current_tasks_index = initialize_task_list_index(self.current_tasks)
        return next_task

    def add_new_tasks(self, tasks: List[str]) -> None:
        for task in tasks:
            if task not in self.current_tasks and task not in self.completed_tasks:
                self.current_tasks.append(Document(task))
        self.current_tasks_index = initialize_task_list_index(self.current_tasks)

    def add_completed_task(self, task: str, result: str) -> None:
        document = Document(f"Task: {task}\nResult: {result}\n")
        self.completed_tasks.append(document)
        self.completed_tasks_index = initialize_task_list_index(self.completed_tasks)
