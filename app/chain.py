from langchain.chains.llm import LLMChain

from .llm import create_llama_chat


def create_chain() -> LLMChain:
    system_prompt = """
    당신은 전문적이고 친절한 AI 어시스턴트입니다. 
    항상 정확하고 도움이 되는 답변을 제공하도록 노력하세요.
    모르는 것에 대해서는 솔직히 모른다고 답변하세요.
    """

    chat_model, prompt_template = create_llama_chat(
        system_prompt=system_prompt,
        model_name="llama3.2:3b",
        model_temperature=0.1,
    )

    chain = LLMChain(llm=chat_model, prompt=prompt_template, verbose=True)

    return chain
