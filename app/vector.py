import json
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

qdrant_client = QdrantClient(host="localhost", port=6333, timeout=60)

# 임베딩 모델 설정
ollama_embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434", model="llama3.2:3b"
)

collection_name = "fruits"


class QdrantVector:
    @staticmethod
    def create_collection() -> bool:
        try:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=3072, distance=models.Distance.COSINE
                ),
            )
        except Exception as e:
            print(f"컬렉션 생성 중 오류 발생: {e}")
            return False

        return True

    @staticmethod
    def delete_collection() -> bool:
        try:
            qdrant_client.delete_collection(collection_name=collection_name)
        except Exception as e:
            print(f"컬렉션 삭제 중 오류 발생: {e}")
            return False

        return True

    @staticmethod
    def text_embedding() -> bool:
        sample_data: list[dict[str, Any]] = json_reader(Path("../sample/dataset.json"))
        documents: list[Document] = [
            Document(
                page_content=sample["description"],
                metadata={
                    key: value for key, value in sample.items() if key != "description"
                },
            )
            for sample in sample_data
        ]

        try:
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=ollama_embeddings,
            )

            vector_store.add_documents(documents)
        except Exception as e:
            print(f"텍스트 임베딩 추가 중 오류 발생: {e}")
            return False

        return True

    @staticmethod
    def force_index() -> bool:
        try:
            qdrant_client.update_collection(
                collection_name=collection_name,
                optimizer_config=models.OptimizersConfigDiff(indexing_threshold=1),
            )
            return True
        except Exception as e:
            print(f"인덱싱 중 오류 발생: {e}")
            return False


def json_reader(file_path: Path) -> list[dict[str, Any]]:
    with open(file_path) as f:
        data = json.load(f)
    return data
