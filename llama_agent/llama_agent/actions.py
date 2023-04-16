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
from langchain.chat_models import ChatOpenAI


def parse_command(response: Response):
    return response.command.action, response.command.args


def run_command(user_query: str, command: str, args: Dict) -> str:
    if command == "search":
        search_terms = args["search_terms"]
        results = search_web(search_terms)
        return analyze_search_results(user_query, search_terms, results)
    elif command == "download":
        url = args["url"]
        doc_name = args["doc_name"]
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
                web_summary = download_web(url[i], doc_name[i])
                results.append(format_web_download(url[i], doc_name[i], web_summary))
                web_summary_cache[doc_name[i]] = web_summary
            print("Writing web summary cache to file")

            with open("data/web_summary_cache.json", "w") as f:
                json.dump(web_summary_cache, f)
            return "\n".join(results)
        else:
            if os.path.exists("data/web_summary_cache.json"):
                with open("data/web_summary_cache.json", "r") as f:
                    web_summary_cache = json.load(f)
            else:
                web_summary_cache = {}
            web_summary = download_web(url, doc_name)
            web_summary_cache[doc_name] = web_summary
            print("Writing web summary cache to file")

            with open("data/web_summary_cache.json", "w") as f:
                json.dump(web_summary_cache, f)
            return format_web_download(url, doc_name, web_summary)
    elif command == "query":
        return query_docs(args["docs"], args["query"])
    elif command == "write":
        return write_to_file(args["file_name"], args["data"])
    elif command == "exit":
        return "exit"
    else:
        raise ValueError(f"Unknown command: {command}")


def search_web(search_terms, max_results=5):
    """Search the Web and obtain a list of web results."""
    results = ddg(search_terms, max_results=max_results)
    return results


def analyze_search_results(user_query, search_terms, results):
    """Analyze the results of the search using llm."""
    doc = Document(json.dumps(results))
    index = GPTListIndex.from_documents([doc])
    response = index.query(
        SEARCH_RESULTS_TEMPLATE.format(search_terms=search_terms, user_query=user_query)
    )
    return response.response


def download_web(url, doc_name):
    """Download the html of the url and save a reference under doc_name.
    Return the summary of the web page.
    """
    reader = BeautifulSoupWebReader()
    docs = reader.load_data([url])
    index = GPTListIndex.from_documents(docs)
    if not os.path.exists("data"):
        os.mkdir("data")
    index.save_to_disk("data/" + doc_name + ".json")
    summary = index.query(
        "Summarize the contents of this web page.", response_mode="tree_summarize"
    )
    return summary.response


def query_docs(docs, query):
    llm_predictor_chatgpt = LLMPredictor(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    )
    service_context_chatgpt = ServiceContext.from_defaults(
        llm_predictor=llm_predictor_chatgpt
    )
    query_configs = [
        {
            "index_struct_type": "simple_dict",
            "query_mode": "default",
            "query_kwargs": {"similarity_top_k": 1},
        },
        {
            "index_struct_type": "keyword_table",
            "query_mode": "simple",
            "query_kwargs": {"response_mode": "tree_summarize"},
        },
    ]
    print("Opening web summary cache")
    with open("data/web_summary_cache.json", "r") as f:
        doc_summary_cache = json.load(f)
    if isinstance(docs, list):
        indices = []
        for doc_name in docs:
            index = GPTListIndex.load_from_disk("data/" + doc_name + ".json")
            indices.append((index, doc_summary_cache[doc_name]))
        graph = ComposableGraph.from_indices(
            GPTListIndex,
            [index[0] for index in indices],
            index_summaries=[index[1] for index in indices],
        )
        response = graph.query(
            query, query_configs=query_configs, service_context=service_context_chatgpt
        )
        return response.response
    else:
        index = GPTListIndex.load_from_disk("data/" + docs + ".json")
        response = index.query(query, service_context=service_context_chatgpt)
        return response.response


def write_to_file(file_name, data):
    print("Writing to file" + file_name)
    with open(file_name, "w") as f:
        f.write(data)
    return "Done"
