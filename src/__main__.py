# from "/goinfre/mait-mou/call_me_maybe/llm_sdk/llm_sdk" import Small_LLM_Model
from llm_sdk import Small_LLM_Model
import argparse
from .data_loader import load_data, load_vocab, LoadError
from .constrained_decoding import get_fn_name, get_parames

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
        "Example of use:\n"
        "Request : What is the sum of -2.77 and 3?\n"
        'function: fn_add_numbers, Prameters : {"a" : -2.77, "b" : 3}'
        "If prompt says '-2', output -2.0. "
        "CRITICAL: When a parameter is a regular expression pattern, "
        "output a complete, syntactically valid regex. "
        "Ensure every '(' has a matching ')' and every '[' has a matching ']'. "
        "Use standard regex syntax (e.g. \\d for digits, \\w for word characters, "
        "\\s for whitespace). Do not truncate the pattern early. "
        'Output only valid JSON: {"name": "<fn>", "parameters": {<args>}}'
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
        exit(1)

    fn_names = set(fn.name for fn in func_defs)
    fn_names_ids = []

    for fn in fn_names:
        fn_names_ids.extend(llm.encode(fn).tolist()[0])

    fn_names_ids = set(fn_names_ids)

    numbers_ids = []

    for i in "0123456789.,−-":
        numbers_ids += llm.encode(i).tolist()[0]

    for p in promts:
        full_prompt = system_prompt(func_defs, p.prompt)
        prompt_ids = llm.encode(full_prompt).tolist()[0]

        result = llm.encode('{"name":"').tolist()[0]

        fn_name = get_fn_name(llm, result, fn_names_ids, fn_names, prompt_ids)

        fn_parames = [fn.parameters for fn in func_defs if fn.name == fn_name][0]

        parames = get_parames(llm, result, fn_parames, numbers_ids, prompt_ids, vocab)

        print(parames)
          
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit()
