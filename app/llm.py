from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama


def create_llama_chat(
    system_prompt: str,
    model_name: str = "llama3.2:3b",
    model_temperature: float = 0.7,
) -> tuple[ChatOllama, ChatPromptTemplate]:
    chat_model = ChatOllama(model=model_name, temperature=model_temperature)
    prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        messages=[("system", system_prompt), ("human", "{input}")]
    )

    return chat_model, prompt_template
