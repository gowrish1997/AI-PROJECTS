import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
print("Root dir added to path:", root_dir)
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
from scripts import base_tools, utils,prompts
load_dotenv()

checkpointer=InMemorySaver()
tools=None

class ChatRequest(BaseModel):
    query: str=Field(..., min_length=1, description="The prompt for the agent to process")
    model:str="gpt-4o-mini"
    thread_id:str="default"

async def get_tools():
    mcp_config = utils.load_mcp_config("gmail", "yahoo-finance", "google-sheets")
    # print("mcp config loaded:", mcp_config)

    client = MultiServerMCPClient(mcp_config)

    mcp_tools = await client.get_tools()

    tools = mcp_tools + [base_tools.web_search, base_tools.get_weather]

    # # Filter tools that work with Gemini
    filter_tools = ['delete_email', 'batch_modify_emails', 'batch_delete_emails','delete_label','delete_filter', 'update_cells']
    
    safe_tools = [tool for tool in tools if tool.name not in filter_tools]

    return safe_tools

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tools
    tools = await get_tools()
    yield

async def stream_agent_response(query, model_name, thread_id):
    llm = ChatOpenAI(model=model_name, temperature=0.2)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompts.get_assistant_prompt(),
        checkpointer=checkpointer
    )
    config={"configurable":{"thread_id":thread_id}}
    async for chunk,metadata in agent.astream({"messages":[HumanMessage(content=query)]},stream_mode="messages", config=config):
        data={
            "type":chunk.__class__.__name__,
            "content": chunk.text,
        }
        if isinstance(chunk, AIMessageChunk) and chunk.tool_calls:
            data["tool_calls"]=chunk.text
        yield (json.dumps(data) + "\n").encode()    

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Agent Stream Server is running."}


@app.post("/chat_stream")
async def chat(request: ChatRequest):
    try:
        return StreamingResponse(stream_agent_response(request.query, request.model, request.thread_id), media_type="application/x-ndjson")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
