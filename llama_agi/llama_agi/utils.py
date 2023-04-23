from typing import Any, List, Optional

from llama_index import GPTSimpleVectorIndex, GPTListIndex, ServiceContext, Document
from llama_index.indices.base import BaseGPTIndex


def initialize_task_list_index(
    documents: List[Document], service_context: Optional[ServiceContext] = None
) -> BaseGPTIndex[Any]:
    return GPTListIndex.from_documents(documents, service_context=service_context)


def initialize_search_index(
    documents: List[Document], service_context: Optional[ServiceContext] = None
) -> BaseGPTIndex[Any]:
    return GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )


def log_current_status(
    cur_task: str, result: str, completed_tasks_summary: str, task_list: List[Document], return_str = False
) -> Optional[str]:
    status_string = f"""
    __________________________________
    Completed Tasks Summary: {completed_tasks_summary.strip()}
    Current Task: {cur_task.strip()}
    Result: {result.strip()}
    Task List: {", ".join([x.get_text().strip() for x in task_list])}
    __________________________________
    """
    if return_str:
        return status_string
    else:
        print(status_string, flush=True)
