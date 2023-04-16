from pydantic import BaseModel, Field, root_validator
from typing import Dict, Union, List


class Command(BaseModel):
    action: str = Field(description="This is the current action")
    args: Dict = Field(description="This is the command's arguments")

    @root_validator
    def validate_all(cls, values):
        # print(f"{values}")
        if values["action"] == "search" and "search_terms" not in values["args"]:
            raise ValueError("malformed search args")
        if values["action"] == "download" and (
            "url" not in values["args"] or "doc_name" not in values["args"]
        ):
            raise ValueError("malformed download args")
        if values["action"] == "query" and (
            "docs" not in values["args"] or "query" not in values["args"]
        ):
            raise ValueError("malformed query args")
        if values["action"] == "write" and (
            "file_name" not in values["args"] or "data" not in values["args"]
        ):
            raise ValueError("malformed write args")
        return values


class Response(BaseModel):
    thoughts: str = Field(description="This what the AI is currently thinking.")
    remember: str = Field(description="This is what the AI just did.")
    reasoning: str = Field(
        description="This is why the AI thinks it will help lead to the user's desired result"
    )
    plan: Union[str, object] = Field(
        description="This is the AI's current plan of action"
    )
    criticism: str = Field(
        description="The AI's constructive self criticism."
    )
    command: Command = Field(description="This is the AI's current command")
