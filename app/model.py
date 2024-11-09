from pydantic import BaseModel


class UserQuestion(BaseModel):
    input: str
