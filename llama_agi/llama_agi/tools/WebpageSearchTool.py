from langchain.agents import tool
from llama_index import download_loader, ServiceContext

from llama_agi.utils import initialize_search_index

BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")


@tool("Search Webpage")
def search_webpage(prompt: str) -> str:
    """Useful for searching a specific webpage. The input to the tool should be URL and query, separated by a newline."""
    loader = BeautifulSoupWebReader()
    if len(prompt.split("\n")) < 2:
        return "The input to search_webpage should be a URL and a query, separated by a newline."

    url = prompt.split("\n")[0]
    query_str = " ".join(prompt.split("\n")[1:])

    try:
        documents = loader.load_data(urls=[url])
        service_context = ServiceContext.from_defaults(chunk_size_limit=512)
        index = initialize_search_index(documents, service_context=service_context)
        query_result = index.as_query_engine(similarity_top_k=3).query(query_str)
        return str(query_result)
    except ValueError as e:
        return str(e)
    except Exception:
        return "Encountered an error while searching the webpage."
