from .models import FunctionDefinition, Prompt
from pydantic import ValidationError
import json

class LoadError(Exception):
    pass


def load_data(path, is_prompt=True):
    try:
        with open(path, "r") as f:
            data = json.load(f)

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
    except ValidationError as e:
        err = e.errors()[0]
        raise LoadError(
            f"Load Error: Fiels: {err['loc'][0]} | Message: {err['msg']}."
        )
