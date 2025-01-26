import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import load_prompt


def pred(example: dict[str, str], model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": example["input"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # prompt += "<|im_start|>assistant\n"
    print(prompt)
    token_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    )
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=1200,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
    )
    print(output[:100])
    return output

