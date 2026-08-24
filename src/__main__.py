"""Entry point for the call_me_maybe function-calling tool."""

from __future__ import annotations

import argparse
from typing import Any

from llm_sdk import Small_LLM_Model

from .build_promp_output import build_output, system_prompt
from .constrained_decoding import get_fn_name, get_parames
from .data_loader import LoadError, load_data, load_vocab


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parse = argparse.ArgumentParser(
        description="call me maybe"
    )

    parse.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json"
    )

    parse.add_argument(
        "--function_definition",
        type=str,
        default="data/input/functions_definition.json"
    )

    parse.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json"
    )

    parse.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B"
    )

    return parse.parse_args()


def main() -> None:
    """Load inputs, run constrained decoding, and write results."""
    args = parse_args()
    llm = Small_LLM_Model()

    try:
        func_defs = load_data(args.function_definition, False)
        promts = load_data(args.input)

        vocab_path = llm.get_path_to_vocab_file()
        vocab = load_vocab(vocab_path)
    except LoadError as err:
        print(err)
        exit(1)

    fn_names = set(fn.name for fn in func_defs)
    fn_names_ids_list: list[int] = []

    for fn in fn_names:
        fn_names_ids_list.extend(llm.encode(fn).tolist()[0])

    fn_names_ids: set[int] = set(fn_names_ids_list)

    numbers_ids: list[int] = []

    for i in "0123456789.,−-":
        numbers_ids += llm.encode(i).tolist()[0]

    result_data: list[dict[str, Any]] = []
    for p in promts:
        full_prompt = system_prompt(func_defs, p.prompt)
        prompt_ids = llm.encode(full_prompt).tolist()[0]

        result = llm.encode('{"name":"').tolist()[0]

        fn_name = get_fn_name(
            llm, result, fn_names_ids, fn_names, prompt_ids
        )

        fn_parames = [
            fn.parameters for fn in func_defs if fn.name == fn_name
        ][0]

        parames = get_parames(
            llm, result, fn_parames, numbers_ids, prompt_ids, vocab
        )

        result_plan = {
            "prompt": p.prompt,
            "name": fn_name,
            "parameters": parames
        }

        result_data.append(result_plan)

    build_output(args.output, result_data)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
