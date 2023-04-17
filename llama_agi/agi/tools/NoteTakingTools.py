from langchain.agents import tool
from llama_index import Document
from agi.utils import initialize_search_index

note_index = initialize_search_index([])


@tool
def record_note(note: str) -> str:
    """Useful for when you need to record a note or reminder for yourself to reference in the future."""
    global note_index
    note_index.insert(Document(note))
    return "Note successfully recorded."


@tool
def search_notes(query_str: str) -> str:
    """Useful for searching through notes that you previously recorded."""
    global note_index
    response = note_index.query(query_str, similarity_top_k=3, response_mode="compact")
    return str(response)
