from pydantic import BaseModel, Field, root_validator
from typing import Dict, Union, List
import json


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

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class Response(BaseModel):
    remember: str = Field(description="This is what the AI just accomplished. Probably should not do it again")
    thoughts: str = Field(description="This what the AI is currently thinking.")
    reasoning: str = Field(
        description="This is why the AI thinks it will help lead to the user's desired result"
    )
    plan: Union[str, object] = Field(
        description="This is the AI's current plan of action"
    )
    command: Command = Field(description="This is the AI's current command")
