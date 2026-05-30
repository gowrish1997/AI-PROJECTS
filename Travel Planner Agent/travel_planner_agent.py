import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from scripts import base_tools, utils,prompts


load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

async def get_tools():
    mcp_config=utils.load_mcp_config("airbnb","google-calendar")
    mcp_client = MultiServerMCPClient(mcp_config)
    mcp_tools = await mcp_client.get_tools()
    tools=mcp_tools+[base_tools.get_weather]
    return tools

async def plan_trip(query,thread_id="default"):
    config={"configurable":{"thread_id":thread_id}}
    system_prompt=prompts.get_travel_planner_prompt()
    memory = InMemorySaver()
    tools=await get_tools()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=memory,
    )

    response = await agent.ainvoke({"messages":[HumanMessage(content=query)]}, config=config)

    print(response)
async def main():
    while True:
        print("\n\n\nAsk Question. Type 'q' or 'quite' to exit.")
        query = input("You: ").strip()

        if query.lower() in ["q", "quite"]:
            print("Exiting chat mode.")
            break

        await plan_trip(query)

if __name__ == "__main__":
    asyncio.run(main())

