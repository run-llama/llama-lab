"""Run Llama conversation agents.

The goal of this is to simulate conversation between two agents.

"""

from llama_index import (
    GPTSimpleVectorIndex, GPTListIndex, Document, ServiceContext
)
from llama_index.indices.base import BaseGPTIndex
from llama_index.data_structs import Node
from llama_index.prompts.prompts import QuestionAnswerPrompt
from collections import deque
from pydantic import BaseModel, Field
from typing import Optional, Dict


def format_text(text: str, user: str) -> str:
    return user + ": " + text


DEFAULT_USER_PREFIX_TMPL = (
    "Your name is {name}. "
    "We provide conversation context between you and other users below. "\
    "You are on a date with someone else. \n"
    # "The user is the plaintiff and the other user is the defendant."
)
DEFAULT_PROMPT_TMPL = (
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information, perform the following task.\n"
    "Task: {query_str}\n"
    "You: "
    # "Here's an example:\n"
    # "Previous line: Hi Bob, good to meet you!\n"
    # "You: Good to meet you too!\n\n"
    # "Previous line: {query_str}\n"
    # "You: "
)
DEFAULT_PROMPT = QuestionAnswerPrompt(DEFAULT_PROMPT_TMPL)



class ConvoAgent(BaseModel):
    """Basic abstraction for a conversation agent."""
    name: str
    st_memory: deque
    lt_memory: BaseGPTIndex
    lt_memory_query_kwargs: Dict = Field(default_factory=dict)
    service_context: ServiceContext
    st_memory_size: int = 10
    # qa_prompt: QuestionAnswerPrompt = DEFAULT_PROMPT
    user_prefix_tmpl: str = DEFAULT_USER_PREFIX_TMPL
    qa_prompt_tmpl: str = DEFAULT_PROMPT_TMPL
    
    class Config:
        arbitrary_types_allowed = True
        
    @classmethod
    def from_defaults(
        cls,
        name: Optional[str] = None,
        st_memory: Optional[deque] = None,
        lt_memory: Optional[BaseGPTIndex] = None,
        service_context: Optional[ServiceContext] = None,
        **kwargs
    ) -> "ConvoAgent":
        name = name or "Agent"
        st_memory = st_memory or deque()
        lt_memory = lt_memory or GPTSimpleVectorIndex([])
        service_context = service_context or ServiceContext.from_defaults()
        return cls(
            name=name,
            st_memory=st_memory,
            lt_memory=lt_memory,
            service_context=service_context,
            **kwargs
        )
                      
    
    def add_message(self, message: str, user: str) -> None:
        """Add message from another user."""
        fmt_message = format_text(message, user)
        self.st_memory.append(fmt_message) 
        while len(self.st_memory) > self.st_memory_size:
            self.st_memory.popleft()
        self.lt_memory.insert(Document(fmt_message))
    
    def generate_message(self, prev_message: Optional[str] = None) -> str:
        """Generate a new message."""
        # if prev_message is None, get previous message using short-term memory
        if prev_message is None:
            prev_message = self.st_memory[-1]

        st_memory_text = "\n".join([l for l in self.st_memory])
        summary_response = self.lt_memory.query(
            f"Tell me a bit more about any context that's relevant "
            f"to the current messages: \n{st_memory_text}",
            # similarity_top_k=10,
            response_mode="compact",
            **self.lt_memory_query_kwargs
        )
        
        # add both the long-term memory summary and the short-term conversation
        list_builder = GPTListIndex([])
        list_builder.insert_nodes([Node(str(summary_response))])
        list_builder.insert_nodes([Node(st_memory_text)])
        
        # question-answer prompt
        full_qa_prompt_tmpl = (
            self.user_prefix_tmpl.format(name=self.name) + "\n" +
            self.qa_prompt_tmpl
        )
        qa_prompt = QuestionAnswerPrompt(full_qa_prompt_tmpl)
        
        response = list_builder.query(
            "Generate the next message in the conversation.", 
            text_qa_template=qa_prompt, 
            response_mode="compact"
        )    
        return str(response)
