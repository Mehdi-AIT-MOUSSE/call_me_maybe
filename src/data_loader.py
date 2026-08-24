"""Load and validate JSON input files and vocabulary data."""

import json
from typing import Any

from pydantic import ValidationError

from .models import FunctionDefinition, Prompt


class LoadError(Exception):
    """Raised when an input file cannot be loaded or validated."""


def load_data(
    path: str,
    is_prompt: bool = True,
) -> list[Prompt] | list[FunctionDefinition]:
    """Load prompts or function definitions from a JSON file.

    Args:
        path: Path to the JSON input file.
        is_prompt: If True, parse entries as Prompt objects;
            otherwise parse as FunctionDefinition objects.

    Returns:
        A list of validated Prompt or FunctionDefinition instances.

    Raises:
        LoadError: If the file is missing, invalid JSON, empty,
            or fails schema validation.
    """
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        if is_prompt:
            data = [Prompt(**prompt) for prompt in data]
        else:
            data = [FunctionDefinition(**func) for func in data]

        if not data:
            raise LoadError(
                "Load Error: Empty data"
            )
        return data
    except json.JSONDecodeError:
        raise LoadError(
            "Load Error: The file is not a valid JSON document."
        )
    except FileNotFoundError:
        raise LoadError(
            "Load Error: The specified file could not be found."
        )
    except PermissionError:
        raise LoadError(
            "Load Error: No permision to read the file."
        )
    except ValidationError as error:
        err = error.errors()[0]
        raise LoadError(
            f"Load Error: Fiels: {err['loc'][0]} | Message: {err['msg']}."
        )


def load_vocab(vocab_path: str) -> dict[str, Any]:
    """Load the tokenizer vocabulary mapping from a JSON file.

    Args:
        vocab_path: Path to the vocabulary JSON file.

    Returns:
        A dictionary mapping token strings to token IDs.

    Raises:
        LoadError: If the file is missing, invalid JSON,
            or cannot be read.
    """
    try:
        with open(vocab_path, "r", encoding="utf-8") as file_handle:
            vocab = json.load(file_handle)

        return vocab

    except json.JSONDecodeError:
        raise LoadError(
            "Load Error: The file is not a valid JSON document."
        )

    except FileNotFoundError:
        raise LoadError(
            "Load Error: The specified file could not be found."
        )

    except PermissionError:
        raise LoadError(
            "Load Error: No permision to read the file."
        )
