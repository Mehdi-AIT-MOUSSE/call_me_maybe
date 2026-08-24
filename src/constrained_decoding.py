"""Constrained decoding for function name and parameter generation."""

from __future__ import annotations

from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from .models import ParamType


def get_fn_name(
    llm: Small_LLM_Model,
    result: list[int],
    fn_names_ids: set[int],
    fn_names: set[str],
    prompt_ids: list[int],
) -> str:
    """Generate a function name using constrained token selection.

    Args:
        llm: The language model instance.
        result: Mutable list of generated token IDs so far.
        fn_names_ids: Token IDs allowed during name generation.
        fn_names: Set of valid function names.
        prompt_ids: Encoded prompt token IDs.

    Returns:
        The selected function name string.
    """
    fn_name = ""

    for _ in range(20):
        logits = np.array(llm.get_logits_from_input_ids(prompt_ids + result))
        mask_logits = np.full(len(logits), -np.inf)

        for token_id in fn_names_ids:
            mask_logits[token_id] = logits[token_id]

        next_token = int(np.argmax(mask_logits))

        result.append(next_token)

        fn_name += llm.decode(next_token)
        if fn_name in fn_names:
            break

    return fn_name


def get_parames(
    llm: Small_LLM_Model,
    result: list[int],
    fn_parames: dict[str, ParamType],
    numbers_ids: list[int],
    prompt_ids: list[int],
    vocab: dict[str, Any],
) -> dict[str, Any]:
    """Generate function parameters using type-aware constrained decoding.

    Args:
        llm: The language model instance.
        result: Mutable list of generated token IDs so far.
        fn_parames: Parameter schema for the selected function.
        numbers_ids: Token IDs allowed for numeric values.
        prompt_ids: Encoded prompt token IDs.
        vocab: Tokenizer vocabulary mapping.

    Returns:
        A dictionary of parameter names to decoded values.
    """
    inject = llm.encode('", "parameters":{').tolist()[0]
    result += inject

    parames: dict[str, Any] = {}
    for i, p in enumerate(fn_parames):
        param_type = fn_parames[p].type
        if param_type == "string":
            inject = llm.encode(f'"{p}":"').tolist()[0]
        else:
            inject = llm.encode(f'"{p}":').tolist()[0]

        result += inject
        param: list[int] = []
        max_tokens = 100
        for step in range(max_tokens):
            logits = np.array(
                llm.get_logits_from_input_ids(prompt_ids + result + param)
            )

            if param_type == "number":
                mask_logits = np.full(len(logits), -np.inf)
                for token_id in numbers_ids:
                    mask_logits[token_id] = logits[token_id]

                delimiter = ','

            else:
                # 293
                diff = len(logits) - len(vocab)
                logits[-diff:] = [-np.inf] * diff
                mask_logits = logits

                delimiter = '"'

            next_token = np.argmax(mask_logits)

            decoded_token = llm.decode([next_token])

            if delimiter in decoded_token or step == max_tokens - 1:
                prefix = decoded_token.split(delimiter)[0]

                if prefix:
                    param += llm.encode(prefix).tolist()[0]

                parames[p] = llm.decode(param)
                if param_type == "number":
                    parames[p] = float(parames[p])

                break

            param.append(int(next_token))

        result += param
        if i != len(fn_parames) - 1:
            if param_type == "string":
                result += llm.encode('",').tolist()[0]
            else:
                result += llm.encode(',').tolist()[0]
        else:
            if param_type == "string":
                result += llm.encode('"}').tolist()[0]
            else:
                result += llm.encode('}').tolist()[0]

    result += llm.encode("}").tolist()[0]

    return parames
