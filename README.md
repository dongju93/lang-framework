# LangChain

## 목표
#### ✅ langchain 을 통해 LocalLLM 에 Prompt 를 주입하고 Chat 을 수행
#### ❌ Vector Storage (ChromaDB) 에 장문의 Text 정보를 Splitter 로 저장
#### ❌ 사용자 질의 시 Vector Storage 를 Retriever 하여 저장된 정보 기반으로 응답
#### ❌ langgraph 를 통해 LLM 응답이 특정 수준 혹은 잘못된 정보가 포함되어 있는지 검토 하여 답변을 재생성(추론)
#### ❌ 위 과정을 langsmith 로 시각화