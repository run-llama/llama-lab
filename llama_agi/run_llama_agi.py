import logging
import sys
import argparse
import json
import time

from agi.ExecutionAgent import SimpleExecutionAgent, ToolExecutionAgent
from agi.TaskManager import TaskManager
from agi.utils import log_current_status


def run_llama_agi(objective: str, initial_task: str, sleep_time: int) -> None:
    task_manager = TaskManager([initial_task])
    simple_execution_agent = SimpleExecutionAgent()
    tool_execution_agent = ToolExecutionAgent()

    # get initial list of tasks
    initial_completed_tasks_summary = task_manager.get_completed_tasks_summary()
    initial_task_prompt = initial_task + "\nReturn the list as an array."
    initial_task_list_str = simple_execution_agent.execute_task(
        objective, initial_task_prompt, initial_completed_tasks_summary
    )
    initial_task_list = task_manager.parse_task_list(initial_task_list_str)

    # add tasks to the task manager
    task_manager.add_new_tasks(initial_task_list)

    # prioritize initial tasks
    task_manager.prioritize_tasks(objective)

    completed_tasks_summary = initial_completed_tasks_summary
    while True:
        # Get the next task
        cur_task = task_manager.get_next_task()

        # Execute current task
        result = tool_execution_agent.execute_task(
            objective, cur_task, completed_tasks_summary
        )

        # store the task and result as completed
        task_manager.add_completed_task(cur_task, result)

        # generate new task(s), if needed
        task_manager.generate_new_tasks(objective, cur_task, result)

        # Summarize completed tasks
        completed_tasks_summary = task_manager.get_completed_tasks_summary()

        # log state of AGI to terminal
        log_current_status(
            cur_task, result, completed_tasks_summary, task_manager.current_tasks
        )

        # Quit the loop?
        if len(task_manager.current_tasks) == 0:
            print("Out of tasks! Objective Accomplished?")
            break

        # wait a bit to let you read what's happening
        time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Llama AGI",
        description="A baby-agi/auto-gpt inspired application, powered by Llama Index!",
    )
    parser.add_argument(
        "-it",
        "--initial-task",
        default="Create a list of tasks",
        help="The initial task for the system to carry out. Default='Create a list of tasks'",
    )
    parser.add_argument(
        "-o",
        "--objective",
        default="Solve world hunger",
        help="The overall objective for the system. Default='Solve world hunger'",
    )
    parser.add_argument(
        "--sleep",
        default=2,
        help="Sleep time (in seconds) between each task loop. Default=2",
        type=int,
    )

    args = parser.parse_args()

    run_llama_agi(args.objective, args.initial_task, args.sleep)
