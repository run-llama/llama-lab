#############################################
##### AGI Prefix #####
#############################################
PREFIX = (
    "You are an autonomous artificial intelligence, capable of planning and executing tasks to achieve an objective.\n"
    "When given an objective, you can plan and execute any number tasks that will help achieve your original objective.\n"
)


#############################################
##### Initial Completed Tasks Summary #####
#############################################
NO_COMPLETED_TASKS_SUMMARY = "You haven't completed any tasks yet."


#############################################
##### Langchain - Execution Agent (Unused Currently) #####
#############################################
LC_PREFIX = PREFIX + "You have access to the following tools:"

LC_FORMAT_INSTRUCTIONS = """Use the following format:
Task: the current task you must complete
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have now completed the task
Final Answer: the final answer to the original input task"""

LC_SUFFIX = (
    "This is your current objective: {objective}\n"
    "Take into account what you have already achieved: {completed_tasks_summary}\n"
    "Using your current objective, your previously completed tasks, and your available tools,"
    "Complete the current task.\n"
    "Begin!\n"
    "Task: {task}\n"
    "Thought: {agent_scratchpad}"
)


#############################################
##### Langchain - Execution Chain #####
#############################################
LC_EXECUTION_PROMPT = (
    "You are an AI who performs one task based on the following objective: {objective}\n."
    "Take into account this summary of previously completed tasks: {completed_tasks_summary}\n."
    "Your task: {task}\n"
    "Response: "
)


#############################################
##### LlamaIndex -- Task Creation #####
#############################################
DEFAULT_TASK_CREATE_TMPL = (
    f"{PREFIX}"
    "Your current objective is as follows: {query_str}\n"
    "Most recently, you completed the task '{prev_task}', which had the result of '{prev_result}'. "
    "A description of your current incomplete tasks are below: \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the current objective, the current incomplete tasks, and the latest completed task, "
    "create new tasks to be completed that do not overlap with incomplete tasks. "
    "Return the tasks as an array."
)
# TASK_CREATE_PROMPT = QuestionAnswerPrompt(DEFAULT_TASK_CREATE_TMPL)

DEFAULT_REFINE_TASK_CREATE_TMPL = (
    f"{PREFIX}"
    "Your current objective is as follows: {query_str}\n"
    "Most recently, you completed the task '{prev_task}', which had the result of '{prev_result}'. "
    "A description of your current incomplete tasks are below: \n"
    "---------------------\n"
    "{context_msg}"
    "\n---------------------\n"
    "Currently, you have created the following new tasks: {existing_answer}"
    "Given the current objective, the current incomplete tasks, list of newly created tasks, and the latest completed task, "
    "add new tasks to be completed that do not overlap with incomplete tasks. "
    "Return the tasks as an array. If you have no more tasks to add, repeat the existing list of new tasks."
)
# REFINE_TASK_CREATE_PROMPT = RefinePrompt(DEFAULT_REFINE_TASK_CREATE_TMPL)


#############################################
##### LlamaIndex -- Task Prioritization #####
#############################################
DEFAULT_TASK_PRIORITIZE_TMPL = (
    f"{PREFIX}"
    "Your current objective is as follows: {query_str}\n"
    "A list of your current incomplete tasks are below: \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the current objective, prioritize the current list of tasks. "
    "Do not remove or add any tasks. Return the results as a numbered list, like:\n"
    "#. First task\n"
    "#. Second task\n"
    "... continue until all tasks are prioritized. "
    "Start the task list with number 1."
)

DEFAULT_REFINE_TASK_PRIORITIZE_TMPL = (
    f"{PREFIX}"
    "Your current objective is as follows: {query_str}\n"
    "A list of additional incomplete tasks are below: \n"
    "---------------------\n"
    "{context_msg}"
    "\n---------------------\n"
    "Currently, you also have the following list of prioritized tasks: {existing_answer}"
    "Given the current objective and existing list, prioritize the current list of tasks. "
    "Do not remove or add any tasks. Return the results as a numbered list, like:\n"
    "#. First task\n"
    "#. Second task\n"
    "... continue until all tasks are prioritized. "
    "Start the task list with number 1."
)
