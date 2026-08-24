"""Pydantic models for function-calling input validation."""

from pydantic import BaseModel


class Prompt(BaseModel):
    """A single natural-language prompt to process."""

    prompt: str


class ParamType(BaseModel):
    """Type descriptor for a function parameter or return value."""

    type: str


class FunctionDefinition(BaseModel):
    """Schema for an available callable function."""

    name: str
    description: str
    parameters: dict[str, ParamType]
    returns: ParamType
