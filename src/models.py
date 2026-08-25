"""Pydantic models for function-calling input validation."""

from typing import Annotated

from pydantic import BaseModel, StringConstraints


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class Prompt(BaseModel):
    """A single natural-language prompt to process."""

    prompt: NonEmptyString


class ParamType(BaseModel):
    """Type descriptor for a function parameter or return value."""

    type: NonEmptyString


class FunctionDefinition(BaseModel):
    """Schema for an available callable function."""

    name: NonEmptyString
    description: NonEmptyString
    parameters: dict[str, ParamType]
    returns: ParamType
