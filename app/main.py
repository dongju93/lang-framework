from fastapi import FastAPI
from langchain.chains.llm import LLMChain
from langserve import add_routes

from .chain import create_chain
from .model import UserQuestion

app = FastAPI()


chain: LLMChain = create_chain()

add_routes(app=app, runnable=chain, path="/chat", input_type=UserQuestion)


# 기본 라우트
@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Chat API"}
