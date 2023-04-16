import datetime
import json

def get_date():
    return datetime.datetime.now().strftime("%Y-%m-%d")


def print_pretty(json_dic):
    if "thoughts" in json_dic:
        print("Thoughts: " + json_dic["thoughts"] + "\n")
    if "reasoning" in json_dic:
        print("Reasoning: " + json_dic["reasoning"] + "\n")
    if "plan" in json_dic:
        print("Plan: " + json.dumps(json_dic["plan"]) + "\n")
    if "command" in json_dic:
        print("Command: " + json.dumps(json_dic["command"]) + "\n")
