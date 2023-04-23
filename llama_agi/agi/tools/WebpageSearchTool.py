from langchain.agents import tool
from llama_index import download_loader

from agi.utils import initialize_search_index

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
        index = initialize_search_index(documents, chunk_size_limit=512)
        query_result = index.query(
            query_str, similarity_top_k=3, response_mode="compact"
        )
        return str(query_result)
    except ValueError as e:
        return str(e)
    except Exception:
        return "Encountered an error while searching the webpage."
