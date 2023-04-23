import argparse
from langchain.agents import load_tools
from langchain.llms import OpenAI

from llama_agi.execution_agent import ToolExecutionAgent
from llama_agi.runners import AutoAGIRunner
from llama_agi.task_manager import LlamaTaskManager
from llama_agi.tools import search_notes, record_note, search_webpage

from llama_index import ServiceContext, LLMPredictor


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
        "--sleep-time",
        default=2,
        help="Sleep time (in seconds) between each task loop. Default=2",
        type=int,
    )

    args = parser.parse_args()

    # LLM setup
    llm = OpenAI(temperature=0, model_name="text-davinci-003")
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm=llm), chunk_size_limit=512
    )

    # llama_agi setup
    task_manager = LlamaTaskManager(
        [args.initial_task], task_service_context=service_context
    )

    tools = load_tools(["google-search-results-json"])
    tools = tools + [search_notes, record_note, search_webpage]
    execution_agent = ToolExecutionAgent(llm=llm, tools=tools)

    # launch the auto runner
    runner = AutoAGIRunner(task_manager, execution_agent)
    runner.run(args.objective, args.initial_task, args.sleep_time)
