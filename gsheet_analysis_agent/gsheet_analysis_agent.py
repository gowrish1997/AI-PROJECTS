import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from scripts import base_tools, utils,prompts


load_dotenv()


# llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.2)
checkpointer=InMemorySaver()

@wrap_tool_call
async def monitor_tool_calls(request,handler):
    try:
        tool_name = getattr(request, 'name', 'unknown')
        print(f"Calling tool: {tool_name}")
        return await handler(request)
        # Log the tool call to LangSmith
    except Exception as e:
        return ToolMessage(content=f"Error during tool call: {str(e)}",tool_call_id=request.tool_call_id)

async def get_tools():
    mcp_config=utils.load_mcp_config("google-sheets","yahoo-finance")
    mcp_client = MultiServerMCPClient(mcp_config)
    mcp_tools = await mcp_client.get_tools()
    tools=mcp_tools+[base_tools.get_weather]
    # Filter tools that work with Gemini
    problematic_tools = ['update_cells']
    safe_tools=[tool for tool in tools if tool.name not in problematic_tools]
    return safe_tools

async def google_sheet_agent(query,thread_id="default"):
     config={"configurable":{"thread_id":thread_id}}
     tools=await get_tools()
     agent=create_agent(
         model=llm,
         tools=tools,
         middleware=[monitor_tool_calls],
         system_prompt=prompts.GOOGLE_SHEETS_PROMPT,
         checkpointer=checkpointer
     )
     result=await agent.ainvoke({"messages":[HumanMessage(content=query)]},config=config)
     return result

async def main():
    while True:
        query=input("Enter query: ")
        response=await google_sheet_agent(query)
        print(response['messages'][-1].text)
        if query.lower() in ["q", "quite"]:
            print("Exiting chat mode.")
            break

        await google_sheet_agent(query)

if __name__ == "__main__":
    asyncio.run(main())