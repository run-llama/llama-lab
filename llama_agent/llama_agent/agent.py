from llama_agent.utils import get_date
from langchain.output_parsers import PydanticOutputParser
from llama_agent.data_models import Response
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage


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
        print(llm_input)
        output: AIMessage = self.llm(llm_input)
        print(output.content)
        self.memory.append(output.content)
        response_obj = parser.parse(output.content)
        print(response_obj)
        return response_obj

    def create_chat_messages(self, desc, task, memory, format_instructions):
        """Create the messages for the agent."""
        messages = []
        template = "{desc}\n{memory}\n{date}\n{format_instructions}"
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)
        messages.append(system_message_prompt)

        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        messages.append(human_message_prompt)

        prompt_template = ChatPromptTemplate.from_messages(messages)

        date_str = "The current date is " + get_date()
        prompt = prompt_template.format_prompt(
            desc=desc,
            memory="\n".join(memory),
            date=date_str,
            format_instructions=format_instructions,
            text=task,
        )

        return prompt
