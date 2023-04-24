# 🤖 Llama AGI 🦙

This python package allows you to quickly create Auto-GPT-like agents, using LlamaIndex and Langchain.

## Setup

Install using pip:

```bash
pip install llama-agi
```

Or install from source:

```bash
git clone https://github.com/run-llama/llama-lab.git
cd llama-lab/llama_agi
pip install -e .
```

## Example Usage

The following shows an example of setting up the `AutoAGIRunner`, which will continue completing tasks (nearly) indefinitely, trying to achieve it's initial objective of "Solve world hunger."

```python
from langchain.agents import load_tools
from langchain.llms import OpenAI

from llama_agi.execution_agent import ToolExecutionAgent
from llama_agi.runners import AutoAGIRunner
from llama_agi.task_manager import LlamaTaskManager
from llama_agi.tools import search_notes, record_note, search_webpage

from llama_index import ServiceContext, LLMPredictor

# LLM setup
llm = OpenAI(temperature=0, model_name='text-davinci-003')
service_context = ServiceContext.from_defaults(llm_predictor=LLMPredictor(llm=llm), chunk_size_limit=512)

# llama_agi setup
task_manager = LlamaTaskManager([args.initial_task], task_service_context=service_context)

tools = load_tools(["google-search-results-json"])
tools = tools + [search_notes, record_note, search_webpage]
execution_agent = ToolExecutionAgent(llm=llm, tools=tools)

# launch the auto runner
runner = AutoAGIRunner(task_manager, execution_agent)
objective = "Solve world hunger"
initial_task = "Create a list of tasks"
sleep_time = 2 
runner.run(objective, initial_task, sleep_time)
```

More examples can be found in the `examples` folder!

## Llama Ecosystem

- LlamaIndex (connecting your LLMs to data): https://github.com/jerryjliu/llama_index
- LlamaHub (community library of data loaders): https://llamahub.ai
