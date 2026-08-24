"""Build LLM prompts and write function-calling results to disk."""

import json
import os
from typing import Any

from .models import FunctionDefinition


def system_prompt(
    func_defs: list[FunctionDefinition],
    user_prompt: str,
) -> str:
    """Build a chat-formatted prompt listing available functions.

    Args:
        func_defs: Available function definitions.
        user_prompt: The user's natural-language request.

    Returns:
        A formatted prompt string for the LLM.
    """
    # function_name(params (type)): descreption . Return number
    available_funcs: list[str] = []
    for func in func_defs:
        params = ", ".join([
            f"{param_name} ({param_type.type})"
            for param_name, param_type in func.parameters.items()
        ])

        available_funcs.append(
            f"{func.name}({params}): {func.description} "
            f"Return {func.returns.type}"
        )

    available_funcs_str = "\n".join(available_funcs)

    system_prompt_text = (
        "You are a function-calling assistant. Given a user request, "
        "select the single most appropriate function from the list below "
        "and provide the correct arguments as JSON.\n\n"
        f"Available functions:\n{available_funcs_str}\n\n"
        "Respond with only the function name and its arguments no "
        "explanation, no extra text. "
        "Example of use:\n"
        "Request : What is the sum of -2.77 and 3?\n"
        'function: fn_add_numbers, Prameters : {"a" : -2.77, "b" : 3}'
        "If prompt says '-2', output -2.0. "
        "CRITICAL: When a parameter is a regular expression pattern, "
        "output a complete, syntactically valid regex. "
        "Ensure every '(' has a matching ')' and every '[' has a "
        "matching ']'. "
        "Use standard regex syntax (e.g. \\d for digits, \\w for word "
        "characters, \\s for whitespace). Do not truncate the pattern "
        "early. "
        'Output only valid JSON: {"name": "<fn>", "parameters": {<args>}}'
    )
    return (
        f"<|im_start|>system\n{system_prompt_text}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_output(output_path: str, ressult_data: list[dict[str, Any]]) -> None:
    """Write function-calling results to a JSON file.

    Args:
        output_path: Destination file path.
        ressult_data: List of result dictionaries to serialize.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(ressult_data, json_file, indent=4)
