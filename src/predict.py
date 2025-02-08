import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def predict(
    input: str,
    system_prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> str:
    token_ids = tokenize(input, system_prompt, tokenizer)

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
    return output


def tokenize(input: str, system_prompt: str, tokenizer: AutoTokenizer) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt"
    )
    return token_ids