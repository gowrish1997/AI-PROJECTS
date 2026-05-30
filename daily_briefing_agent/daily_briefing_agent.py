import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from scripts import base_tools, utils,prompts


load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
checkpointer=InMemorySaver()

async def get_tools():
    mcp_config=utils.load_mcp_config("gmail","google-calendar", "yahoo-finance")
    mcp_client = MultiServerMCPClient(mcp_config)
    mcp_tools = await mcp_client.get_tools()
    tools=mcp_tools+[base_tools.get_weather]
    # # Filter tools that work with Gemini
    filter_tools = ['delete_email', 'batch_modify_emails', 'batch_delete_emails','delete_label','delete_filter']
    
    safe_tools = [tool for tool in tools if tool.name not in filter_tools]
    # print("Available tools:", [tool.name for tool in safe_tools])
    return safe_tools

async def daily_briefing_agent(query,thread_id="default"):
     config={"configurable":{"thread_id":thread_id}}
     tools=await get_tools()
     agent=create_agent(
         model=llm,
         tools=tools,
         system_prompt=prompts.get_daily_briefing_prompt(),
         checkpointer=checkpointer
     )
     result=await agent.ainvoke({"messages":[HumanMessage(content=query)]},config=config)
     print(result['messages'][-1].text)

async def ask():
    while True:
        query=input("Enter query: ")
        if query.lower() in ["q", "quit"]:
            print("Exiting chat mode.")
            break
        await daily_briefing_agent(query)
        

if __name__ == "__main__":
    query = """Give me my daily briefing:
                    1. Today's weather
                    2. Today's calendar events
                   3. Summary of unread emails
                    4. Top news headlines"""
    asyncio.run(ask())