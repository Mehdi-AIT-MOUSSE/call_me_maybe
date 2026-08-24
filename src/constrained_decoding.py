import numpy as np

def get_fn_name(llm, result, fn_names_ids, fn_names, prompt_ids):
    fn_name = ""

    for _ in range(20):
        logits = np.array(llm.get_logits_from_input_ids(prompt_ids + result))
        mask_logits = np.full(len(logits), -np.inf)

        for id in fn_names_ids:
            mask_logits[id] = logits[id]

        next_token = np.argmax(mask_logits)

        result.append(next_token)

        fn_name += llm.decode(next_token)
        if fn_name in fn_names:
            break

    return fn_name

def get_parames(llm, result, fn_parames, numbers_ids, prompt_ids, vocab):
    inject = llm.encode('", "parameters":{').tolist()[0]
    result += inject 

    parames = {}
    for i, p in enumerate(fn_parames):
        type = fn_parames[p].type
        if type == "string":
            inject = llm.encode(f'"{p}":"').tolist()[0]
        else:
            inject = llm.encode(f'"{p}":').tolist()[0]

        result += inject
        param = []
        max_tokens = 100
        for step in range(max_tokens):
            logits = np.array(llm.get_logits_from_input_ids(prompt_ids + result + param))

            if type == "number":
                mask_logits = np.full(len(logits), -np.inf)
                for id in numbers_ids:
                    mask_logits[id] = logits[id]

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
                if type == "number":
                    parames[p] = float(parames[p])

                break

            param.append(next_token)

        result += param
        if i != len(fn_parames)-1:
            if type == "string":
                result += llm.encode('",').tolist()[0]
            else:
                result += llm.encode(',').tolist()[0]
        else:
            if type == "string":
                result += llm.encode('"}').tolist()[0]
            else:
                result += llm.encode('}').tolist()[0]


    result += llm.encode("}").tolist()[0]

    return parames