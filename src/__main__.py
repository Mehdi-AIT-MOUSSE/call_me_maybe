# from "/goinfre/mait-mou/call_me_maybe/llm_sdk/llm_sdk" import Small_LLM_Model
from llm_sdk import Small_LLM_Model
import json
import argparse

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


def main():
    args = parse_args()
    print(args.input)

    print(args.function_definition)
    print(args.input)
    print(args.input)

        





if __name__ == "__main__":
    main()




# def encode(self, text: str) -> torch.Tensor:

# def decode(self, ids: torch.Tensor | list[int]) -> str:

# def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]: