# from "/goinfre/mait-mou/call_me_maybe/llm_sdk/llm_sdk" import Small_LLM_Model
from llm_sdk import Small_LLM_Model
import argparse
from .data_loader import load_data, load_vocab, LoadError

import numpy as np

def parse_args():
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

def system_prompt(func_defs, user_prompt):
    # function_name(params (type)): descreption . Return number
    available_funcs = []
    for func in func_defs:
        params = ", ".join([f"{param_name} ({param_type.type})" 
                            for param_name, param_type in func.parameters.items()])

        available_funcs.append(
            f"{func.name}({params}): {func.description} Return {func.returns.type}"
            )

    available_funcs = "\n".join(available_funcs)

    system_prompt = (
        "You are a function-calling assistant. Given a user request, "
        "select the single most appropriate function from the list below "
        "and provide the correct arguments as JSON.\n\n"
        f"Available functions:\n{available_funcs}\n\n"
        "Respond with only the function name and its arguments no "
        "explanation, no extra text. "
        "Output only valid JSON: {'name': '<fn>', 'args': {<args>}}"
    )

    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def main():
    args = parse_args()
    llm = Small_LLM_Model()

    try:
        func_defs = load_data(args.function_definition, False)
        promts = load_data(args.input)

        vocab_path = llm.get_path_to_vocab_file()
        vocab = load_vocab(vocab_path)
    except LoadError as err:
        print(err)

    full_prompt = system_prompt(func_defs, promts[0].prompt)

if __name__ == "__main__":
    main()



# def encode(self, text: str) -> torch.Tensor:

# def decode(self, ids: torch.Tensor | list[int]) -> str:

# def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: