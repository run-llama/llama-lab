import json
from llama_agent.agent import Agent
import llama_agent.const as const
from llama_agent.utils import print_pretty
from llama_agent.actions import run_command
from langchain.chat_models import ChatOpenAI

def main():
    # Enter your OpenAI API key here:

    openaichat = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.0,
        openai_api_key="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",
        openai_organization="org-1ZDAvajC6v2ZtAP9hLEIsXRz",
        max_tokens=400,
    )

    user_query = input("Enter what you would like LlamaAgent to do:\n")
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
            "Should I run the command \""
            + action
            + "\" with args "
            + json.dumps(args)
            + "? (y/[N])\n"
        )
        if user_confirm == "y":
            action_results = run_command(user_query, action, args, openaichat)
            # print(action_results)
            agent.memory.append(action_results)
            if action_results == "exit":
                break
        else:
            break


if __name__ == "__main__":
    main()
