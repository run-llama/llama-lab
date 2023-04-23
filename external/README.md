# External Projects

Llama Lab also contains references to amazing external subprojects using LlamaIndex in novel ways.

### INSIGHT

Insight is an autonomous AI that can do medical research. It has a boss agent that takes an objective and an executive summary of the tasks completed already and their results and creates a task list. A worker agent picks up a task from the list and completes it, saving the results to llama index. The boss gets informed of the results and changes/reprioritizes the task list. The workers can call into the pubmed and mygene APIs (more to come). The workers also get context from llama index to help complete their tasks.

[Repo](https://github.com/oneil512/INSIGHT)