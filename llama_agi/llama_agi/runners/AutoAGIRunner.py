import time
from typing import List, Optional

from llama_agi.runners.base import BaseAGIRunner
from llama_agi.execution_agent.SimpleExecutionAgent import SimpleExecutionAgent
from llama_agi.utils import log_current_status


class AutoAGIRunner(BaseAGIRunner):
    def run(
        self,
        objective: str,
        initial_task: str,
        sleep_time: int,
        initial_task_list: Optional[List[str]] = None,
    ) -> None:
        # get initial list of tasks
        if initial_task_list:
            self.task_manager.add_new_tasks(initial_task_list)
        else:
            initial_completed_tasks_summary = (
                self.task_manager.get_completed_tasks_summary()
            )
            initial_task_prompt = initial_task + "\nReturn the list as an array."

            # create simple execution agent using current agent
            simple_execution_agent = SimpleExecutionAgent(
                llm=self.execution_agent._llm,
                max_tokens=self.execution_agent.max_tokens,
                prompts=self.execution_agent.prompts,
            )
            initial_task_list_result = simple_execution_agent.execute_task(
                objective=objective,
                task=initial_task_prompt,
                completed_tasks_summary=initial_completed_tasks_summary,
            )

            initial_task_list = self.task_manager.parse_task_list(initial_task_list_result['output'])

            # add tasks to the task manager
            self.task_manager.add_new_tasks(initial_task_list)

        # prioritize initial tasks
        self.task_manager.prioritize_tasks(objective)

        completed_tasks_summary = initial_completed_tasks_summary
        while True:
            # Get the next task
            cur_task = self.task_manager.get_next_task()

            # Execute current task
            result = self.execution_agent.execute_task(
                objective=objective,
                cur_task=cur_task,
                completed_tasks_summary=completed_tasks_summary,
            )['output']

            # store the task and result as completed
            self.task_manager.add_completed_task(cur_task, result)

            # generate new task(s), if needed
            self.task_manager.generate_new_tasks(objective, cur_task, result)

            # Summarize completed tasks
            completed_tasks_summary = self.task_manager.get_completed_tasks_summary()

            # log state of AGI to terminal
            log_current_status(
                cur_task,
                result,
                completed_tasks_summary,
                self.task_manager.current_tasks,
            )

            # Quit the loop?
            if len(self.task_manager.current_tasks) == 0:
                print("Out of tasks! Objective Accomplished?")
                break

            # wait a bit to let you read what's happening
            time.sleep(sleep_time)
