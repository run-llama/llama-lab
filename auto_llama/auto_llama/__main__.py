import json
from auto_llama.agent import Agent
import auto_llama.const as const
from auto_llama.utils import print_pretty
from auto_llama.actions import run_command
from langchain.chat_models import ChatOpenAI

import logging


def main():
    logger = logging.getLogger()
    logger.level = logging.WARN
    # Enter your OpenAI API key here:

    import os
    os.environ["OPENAI_API_KEY"] = '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'

    openaichat = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.0,
        max_tokens=400,
    )

    user_query = input("Enter what you would like AutoLlama to do:\n")
    if user_query == "":
        user_query = "Summarize the financial news from the past week."
        print("I will summarize the financial news from the past week.\n")
    agent = Agent(const.DEFAULT_AGENT_PREAMBLE, user_query, openaichat)
    while True:
        print("Thinking...")
        response = agent.get_response()
        print_pretty(response)
        action, args = response.command.action, response.command.args
        user_confirm = input(
            'Should I run the command "'
            + action
            + '" with args '
            + json.dumps(args)
            + "? (y/[N])\n"
        )
        if user_confirm == "y":
            action_results = run_command(user_query, action, args, openaichat)
            # print(action_results)
            agent.memory.append(action_results)
            if action_results == "exit" or action_results == "done":
                break
        else:
            break


if __name__ == "__main__":
    main()
