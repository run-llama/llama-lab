import json
import streamlit as st
import time
from typing import List, Optional

from llama_agi.runners.base import BaseAGIRunner
from llama_agi.execution_agent.SimpleExecutionAgent import SimpleExecutionAgent
from llama_agi.utils import log_current_status


def make_intermediate_steps_pretty(json_str: str) -> List[str]:
    steps = json.loads(json_str)
    output = []
    for action_set in steps:
        for step in action_set:
            if isinstance(step, list):
                output.append(step[-1])
            else:
                output.append(step)
    return output


class AutoStreamlitAGIRunner(BaseAGIRunner):
    def run(
        self,
        objective: str,
        initial_task: str,
        sleep_time: int,
        initial_task_list: Optional[List[str]] = None,
        max_iterations: Optional[int] = None
    ) -> None:
        
        run_initial_task = False
        if 'logs' not in st.session_state:
            st.session_state['logs'] = []
            st.session_state['state_str'] = "No state yet!"
            st.session_state['tasks_summary'] = ""
            run_initial_task = True

        logs_col, state_col = st.columns(2)

        with logs_col:
            st.subheader("Execution Log")
            st_logs = st.empty()
        st_logs.write(st.session_state['logs'])
        
        with state_col:
            st.subheader("AGI State")
            st_state = st.empty()
        st_state.write(st.session_state['state_str'])

        if run_initial_task:
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

            tasks_summary = initial_completed_tasks_summary
            st.session_state['tasks_summary'] = tasks_summary

            # update streamlit state
            st.session_state['state_str'] = log_current_status(initial_task, initial_task_list_result['output'], tasks_summary, self.task_manager.current_tasks, return_str=True)
            if st.session_state['state_str']:
                st_state.markdown(st.session_state['state_str'].replace("\n", "\n\n"))

        for _ in range(0, max_iterations):
            # Get the next task
            cur_task = self.task_manager.get_next_task()

            # Execute current task
            result_dict = self.execution_agent.execute_task(
                objective=objective,
                cur_task=cur_task,
                completed_tasks_summary=st.session_state['tasks_summary'],
            )
            result = result_dict['output']
            
            # update logs 
            log = make_intermediate_steps_pretty(json.dumps(result_dict['intermediate_steps'])) + [result]
            st.session_state['logs'].append(log)
            st_logs.write(st.session_state['logs'])

            # store the task and result as completed
            self.task_manager.add_completed_task(cur_task, result)

            # generate new task(s), if needed
            self.task_manager.generate_new_tasks(objective, cur_task, result)

            # Summarize completed tasks
            completed_tasks_summary = self.task_manager.get_completed_tasks_summary()
            st.session_state['tasks_summary'] = completed_tasks_summary

            # log state of AGI to streamlit
            st.session_state['state_str'] = log_current_status(
                cur_task,
                result,
                completed_tasks_summary,
                self.task_manager.current_tasks,
                return_str=True
            )
            if st.session_state['state_str'] is not None:
                st_state.markdown(st.session_state['state_str'].replace("\n", "\n\n"))

            # Quit the loop?
            if len(self.task_manager.current_tasks) == 0:
                st.success("Out of tasks! Objective Accomplished?")
                break

            # wait a bit to let you read what's happening
            time.sleep(sleep_time)
    
