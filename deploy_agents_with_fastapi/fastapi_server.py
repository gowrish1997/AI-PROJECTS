from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import uvicorn
load_dotenv()


app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str=Field(..., min_length=1, description="The prompt for the agent to process")
    model:str="gpt-4o-mini"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/chat")
async def chat(request: ChatRequest):
   if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty") 
   try:
        model=ChatOpenAI(model=request.model,temperature=0.2)
        agent=create_agent(
            model=model,
            tools=[],
            system_prompt="You are a helpful assistant.",
        )
        response=await agent.ainvoke({"messages":[{"role":"user","content":request.prompt}]})
   except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))     
   return {"response": response['messages'][-1].text}  

@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

