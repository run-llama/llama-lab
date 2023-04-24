# 🦙🧪  Llama Lab 🧬🦙

Llama Lab is a repo dedicated to building cutting-edge projects using [LlamaIndex](https://github.com/jerryjliu/llama_index). 

LlamaIndex is an interface for LLM data augmentation. It provides easy-to-use and flexible tools to index
various types of data. At its core, it can be used to index a knowledge corpus. But it can also be used
to index tasks, and provide memory-like capabilities for any outer agent abstractions.

Here's an overview of some of the amazing projects we're exploring:
- llama_agi (a [babyagi](https://github.com/yoheinakajima/babyagi) and [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) inspired project to create/plan/and solve tasks)
- auto_llama (an [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT) inspired project to search/download/query the Internet to solve user-specified tasks).

Each folder is a stand-alone project. See below for a description of each project along with usage examples.

**Contributing**: We're very open to contributions! This can include the following:
- Extending an existing project
- Create a new Llama Lab project
- Modifying capabilities in the core [LlamaIndex](https://github.com/jerryjliu/llama_index) repo in order to support Llama Lab projects.


## Current Labs

### llama_agi

Inspired from [babyagi](https://github.com/yoheinakajima/babyagi) and [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT), using LlamaIndex as a task manager and LangChain as a task executor.

The current version of this folder will start with an overall objective ("solve world hunger" by default), and create/prioritize the tasks needed to achieve that objective. LlamaIndex is used to create and prioritize tasks, while LangChain is used to guess the "result" of completing each action.

Using LangChain and LlamaIndex, llama_agi has access to the following tools: google-search, webpage reading, and note-taking. Note that the google-search tool requires [a Google API key and a CSE ID](https://cse.google.com/cse/).

This will run in a loop until the task list is empty (or maybe you run out of OpenAI credits 😉).

Example Usage:

```python
cd llama_agi/examples
export OPENAI_API_KEY="key"
export GOOGLE_API_KEY="key"
export GOOGLE_CSE_ID="id"
python ./auto_runner_example.py --initial-task "Create a task list" --objective "Solve world hunger" --sleep 2
```

```bash
python ./run_llama_agi.py -h
usage: Llama AGI [-h] [-it INITIAL_TASK] [-o OBJECTIVE] [--sleep SLEEP]

A baby-agi/auto-gpt inspired application, powered by Llama Index!

options:
  -h, --help            show this help message and exit
  -it INITIAL_TASK, --initial-task INITIAL_TASK
                        The initial task for the system to carry out. Default='Create a list of tasks'
  -o OBJECTIVE, --objective OBJECTIVE
                        The overall objective for the system. Default='Solve World Hunger'
  --sleep SLEEP         Sleep time (in seconds) between each task loop. Default=2
```
### auto_llama

Inspired by [autogpt](https://github.com/Significant-Gravitas/Auto-GPT). This implement its own Agent system similar to AutoGPT. 
Given a user query, this system has the capability to search the web and download web pages, before analyzing the combined data and compiling a final answer to the user's prompt.

Example usage:

```bash
cd auto_llama
pip install -r requirements.txt
python -m auto_llama
Enter what you would like AutoLlama to do:
Summarize the financial news from the past week.

```

### Conversational Agents

This is a fun conversational simulator between different agents. You can choose
to provide some details about the context/setting, and watch as the conversation
between different agents evolves.

A sample notebook is provided in the `convo_agents` folder. Usage:

```bash
cd convo_agents

jupyter notebook ConvoAgents.ipynb
```

### External projects

We also provide references to other project repos using LlamaIndex in novel ways.

These repos are hosted as submodules in our `external` folder.

Check it out here: https://github.com/run-llama/llama-lab/tree/main/external

## Ecosystem

Llama Lab is part of the broader Llama ecosystem.
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [LlamaHub](https://llamahub.ai/) ([repo](https://github.com/emptycrown/llama-hub))

Community:
- [Twitter](https://twitter.com/gpt_index)
- [Discord](https://t.co/3ktq3zzYII)
