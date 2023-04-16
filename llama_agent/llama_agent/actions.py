from duckduckgo_search import ddg
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index import GPTListIndex
from llama_agent.data_models import Response
from typing import Dict


def parse_command(response: Response):
    return response.command.action, response.command.args


def run_command(user_query: str, command: str, args: Dict):
    if command == "search":
        search_terms = args["search_terms"]
        results = search_web(search_terms)
        return analyze_search_results(user_query, search_terms, results)
    elif command == "download":
        url = args["url"]
        doc_name = args["doc_name"]
        web_summary = download_web(url, doc_name)
        analyze_web_download(url, doc_name, web_summary)
    elif command == "query":
        return query_docs(args["docs"], args["query"])
    elif command == "write":
        return write_to_file(args["file_name"], args["data"])
    elif command == "exit":
        return "exit"


def search_web(search_terms, max_results=10):
    """Search the Web and obtain a list of web results."""
    results = ddg(search_terms, max_results=max_results)
    return results


def analyze_search_results(user_query, search_terms, results):
    """Analyze the results of the search using llm."""
    template = (
        "I searched for {search_terms} and found the following results:\n{results}\n"
        + "If any of these results help to answer the user's query {user_query} I should respond with which web urls I should download."
        + "Otherwise I should try with different search terms."
    )


def download_web(url, doc_name):
    """Download the html of the url and save a reference under doc_name.
    Return the summary of the web page.
    """
    reader = BeautifulSoupWebReader()
    docs = reader.load_data([url])
    index = GPTListIndex.from_documents(docs)
    index.save_to_disk("data/" + doc_name + ".json")
    summary = index.query(
        "Summarize the contents of this web page.", response_mode="tree_summarize"
    )
    return summary


def query_docs(docs, query):
    pass


def write_to_file(file_name, data):
    pass
