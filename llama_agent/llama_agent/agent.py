from llama_agent.utils import get_date
from langchain.output_parsers import PydanticOutputParser
from llama_agent.data_models import Response
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage
from typing import List
from llama_agent.tokens import count_tokens


class Agent:
    """A class representing an agent.

    Attributes:
        desc(str):
            A description of the agent used in the preamble.
        task(str):
            The task the agent is supposed to perform.
        memory(list):
            A list of the agent's memories.
        llm(BaseLLM):
            The LLM used by the agent.
    """

    def __init__(
        self,
        desc,
        task,
        llm,
        memory=[],
    ):
        """Initialize the agent."""
        self.desc = desc
        self.task = task
        self.memory = memory
        self.llm = llm

        memory.append("Here is a list of your previous actions:")

    def get_response(self) -> Response:
        """Get the response given the agent's current state."""
        parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Response)
        format_instructions = parser.get_format_instructions()
        llm_input = self.create_chat_messages(
            self.desc, self.task, self.memory, format_instructions
        ).to_messages()
        # print(llm_input)
        output: AIMessage = self.llm(llm_input)
        # print(output.content)
        self.memory.append("Old thought: " + output.content)
        response_obj = parser.parse(output.content)
        # print(response_obj)
        return response_obj

    def create_chat_messages(
        self, desc: str, task: str, memory: List[str], format_instructions: str
    ):
        """Create the messages for the agent."""
        messages = []
        system_template = "{desc}\n{memory}\n{date}\n{format_instructions}"
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template
        )
        messages.append(system_message_prompt)

        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        messages.append(human_message_prompt)

        prompt_template = ChatPromptTemplate.from_messages(messages)

        date_str = "The current date is " + get_date()
        recent_memories = self.create_memories(memory)
        print(recent_memories)
        prompt = prompt_template.format_prompt(
            desc=desc,
            memory=recent_memories,
            date=date_str,
            format_instructions=format_instructions,
            text=task,
        )

        return prompt

    def create_memories(self, memory: List[str], max_tokens: int = 2000):
        # print(memory)
        token_counter = 0
        memories: List[str] = []
        memories.insert(0, memory[0])  # always include memories header.
        token_counter += count_tokens(memory[0])
        memory_index = len(memory) - 1
        while memory_index > 0 and token_counter < max_tokens:
            memories.insert(1, memory[memory_index])
            token_counter += count_tokens(memory[memory_index])
            memory_index -= 1
        return "\n".join(memories)
