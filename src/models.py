from pydantic import BaseModel


class Prompt(BaseModel):
    prompt: str


class ParamType(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType


