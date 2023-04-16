import datetime
import json
from llama_agent.data_models import Response

def get_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")


def print_pretty(response: Response):
    print("Thoughts: " + response.thoughts + "\n")
    print("Remember: " + response.remember + "\n")
    print("Reasoning: " + response.reasoning + "\n")
    print("Plan: " + json.dumps(response.plan) + "\n")
    print("Criticism: " + response.criticism + "\n")
    print("Command: " + json.dumps(response.command) + "\n")
