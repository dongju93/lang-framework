from chain import create_chain
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.runnables import Runnable
from langserve import add_routes
from model import UserQuestion
from vector import QdrantVector

load_dotenv()


app = FastAPI()


chain: Runnable = create_chain()

add_routes(app=app, runnable=chain, path="/chat", input_type=UserQuestion)


# 기본 라우트
@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to the Chat API"}


@app.post("/vector")
async def create_collection():
    if not QdrantVector.create_collection():
        return {"message": "Failed to create vector DB"}
    return {"message": "Successfully created vector DB"}


@app.delete("/vector")
async def delete_collection():
    if not QdrantVector.delete_collection():
        return {"message": "Failed to delete vector DB"}
    return {"message": "Successfully deleted vector DB"}


@app.post("/vector/text")
async def text_embedding():
    if not QdrantVector.text_embedding():
        return {"message": "Failed to add text embeddings"}
    return {"message": "Successfully added text embeddings"}


@app.post("/vector/index")
async def force_index():
    if not QdrantVector.force_index():
        return {"message": "Failed to force index"}
    return {"message": "Successfully forced index"}
