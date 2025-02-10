import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import Dataset

def load_model_and_tokenizer(model_name: str, fast: bool) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer.

    Args:
        model_name (str): Model name.
    Returns:
        tuple: Model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if fast:
        model = LLM(model=model_name, tokenizer=tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
    model.eval()
    return model, tokenizer


def predict(
    dataset: Dataset,
    system_prompt: str,
    model_name: str,
    fast: bool,
) -> str:
    model, tokenizer = load_model_and_tokenizer(model_name, fast)
    results = []
    for example in dataset["test"]:
        prompt = prepare_message(example["input"], system_prompt, tokenizer)
        if fast:
            output = _predict_vllm(prompt, model)
        else:
            output = _predict_huggingface(prompt, model, tokenizer)
        results.append({**example, model_name: output})
    return results


def _predict_huggingface(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> str:
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
    return output


def _predict_vllm(
    prompt: str,
    model: LLM,
) -> str:
    sampling_params = SamplingParams(temperature=0.8)
    return model.generate(prompt, sampling_params)


def prepare_message(user_prompt: str, system_prompt: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt
    