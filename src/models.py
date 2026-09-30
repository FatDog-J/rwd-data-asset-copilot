from pydantic import BaseModel


class RWDAnswer(BaseModel):
    answer: str
    relevant_tables_or_fields: list[str]
    limitations: list[str]
    sources: list[str]
