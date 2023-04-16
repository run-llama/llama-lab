import json
import os

from duckduckgo_search import ddg
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index import GPTListIndex
from llama_agent.data_models import Response
from typing import Dict
from llama_agent.const import SEARCH_RESULTS_TEMPLATE, format_web_download
from llama_index import Document
from llama_index.indices.composability import ComposableGraph
from llama_index import GPTListIndex, LLMPredictor, ServiceContext
from langchain.llms.base import BaseLLM
from llama_index.logger import LlamaLogger


def run_command(user_query: str, command: str, args: Dict, llm: BaseLLM) -> str:
    llama_logger = LlamaLogger()
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(llm), llama_logger=llama_logger
    )
    if command == "search":
        search_terms = args["search_terms"]
        print("Searching...\n")
        results = search_web(search_terms)
        response = analyze_search_results(
            user_query, search_terms, results, service_context
        )
        print(response + "\n")
        return response
    elif command == "download":
        url = args["url"]
        doc_name = args["doc_name"]
        print("Downloading web page...\n")
        if isinstance(url, str) and "[" in url and "]" in url:  # list parsing case
            url = url.strip("[").strip("]").split(", ")
            doc_name = doc_name.strip("[").strip("]").split(", ")
        if isinstance(url, list):
            if len(url) != len(doc_name):
                raise ValueError("url and doc_name must have the same length")
            results = []
            if os.path.exists("data/web_summary_cache.json"):
                with open("data/web_summary_cache.json", "r") as f:
                    web_summary_cache = json.load(f)
            else:
                web_summary_cache = {}
            for i in range(len(url)):
                web_summary = download_web(url[i], doc_name[i], service_context)
                results.append(format_web_download(url[i], doc_name[i], web_summary))
                web_summary_cache[doc_name[i]] = web_summary
            print("Writing web summary cache to file")

            with open("data/web_summary_cache.json", "w") as f:
                json.dump(web_summary_cache, f)
            response = "\n".join(results)
            print(response)
            return response
        else:
            if os.path.exists("data/web_summary_cache.json"):
                with open("data/web_summary_cache.json", "r") as f:
                    web_summary_cache = json.load(f)
            else:
                web_summary_cache = {}
            web_summary = download_web(url, doc_name, service_context)
            web_summary_cache[doc_name] = web_summary
            print("Writing web summary cache to file")

            with open("data/web_summary_cache.json", "w") as f:
                json.dump(web_summary_cache, f)
            response = format_web_download(url, doc_name, web_summary)
            print(response)
            return response
    elif command == "query":
        print("Querying...\n")
        response = query_docs(args["docs"], args["query"], service_context)
        print(response)
        return response
    elif command == "write":
        print("Writing to file...\n")
        return write_to_file(args["file_name"], args["data"])
    elif command == "exit":
        print("Exiting...\n")
        return "exit"
    else:
        raise ValueError(f"Unknown command: {command}")


def search_web(search_terms, max_results=5):
    """Search the Web and obtain a list of web results."""
    results = ddg(search_terms, max_results=max_results)
    return results


def analyze_search_results(user_query, search_terms, results, service_context):
    """Analyze the results of the search using llm."""
    doc = Document(json.dumps(results))
    index = GPTListIndex.from_documents([doc], service_context=service_context)
    response = index.query(
        SEARCH_RESULTS_TEMPLATE.format(search_terms=search_terms, user_query=user_query)
    )
    return response.response


def download_web(url: str, doc_name: str, service_context: ServiceContext):
    """Download the html of the url and save a reference under doc_name.
    Return the summary of the web page.
    """
    reader = BeautifulSoupWebReader()
    docs = reader.load_data([url])
    index = GPTListIndex.from_documents(docs, service_context=service_context)
    if not os.path.exists("data"):
        os.mkdir("data")
    index.save_to_disk("data/" + doc_name + ".json")
    summary = index.query(
        "Summarize the contents of this web page.", response_mode="tree_summarize"
    )
    return summary.response


def query_docs(docs, query, service_context):
    query_configs = [
        {
            "index_struct_type": "list",
            "query_mode": "default",
            "query_kwargs": {"response_mode": "tree_summarize", "use_async": True},
        }
    ]
    print("Opening web summary cache")
    with open("data/web_summary_cache.json", "r") as f:
        doc_summary_cache = json.load(f)
    if isinstance(docs, list):
        indices = []
        for doc_name in docs:
            index = GPTListIndex.load_from_disk(
                "data/" + doc_name + ".json", service_context=service_context
            )
            indices.append((index, doc_summary_cache[doc_name]))
        graph = ComposableGraph.from_indices(
            GPTListIndex,
            [index[0] for index in indices],
            index_summaries=[index[1] for index in indices],
            service_context=service_context,
        )
        response = graph.query(
            query, query_configs=query_configs, service_context=service_context
        )
        return response.response
    else:
        index = GPTListIndex.load_from_disk(
            "data/" + docs + ".json", service_context=service_context
        )
        response = index.query(query, service_context=service_context)
        return response.response


def write_to_file(file_name, data):
    print("Writing to file" + file_name)
    with open(file_name, "w") as f:
        f.write(data)
    return "done"
