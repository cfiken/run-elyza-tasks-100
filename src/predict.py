import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

BATCH_SIZE = 1


def load_model_and_tokenizer(model_name: str, fast: bool) -> tuple[AutoModelForCausalLM | LLM, AutoTokenizer]:
    """Load model and tokenizer.

    Args:
        model_name (str): Model name.
    Returns:
        tuple: Model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if fast:
        model = LLM(
            model=model_name,
            tensor_parallel_size=4,
            dtype="auto",
            gpu_memory_utilization=0.8
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto"
        )
        model.eval()
    return model, tokenizer


def predict(
    df: pd.DataFrame,
    system_prompt: str,
    model_name: str,
    fast: bool,
) -> str:
    model, tokenizer = load_model_and_tokenizer(model_name, fast)
    if fast:
        results = _predict_vllm(df, system_prompt, model, model_name)
    else:
        results = _predict_huggingface(df, system_prompt, model, model_name, tokenizer)
    return results


def _predict_huggingface(
    df: pd.DataFrame,
    system_prompt: str,
    model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
) -> str:
    results = []
    for _, row in df.iterrows():
        prompt = prepare_message(row["input"] + "\n\n" + row["context"], system_prompt, tokenizer)
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
        results.append({**row, model_name: output})
    return results


def _predict_vllm(
    df: pd.DataFrame,
    system_prompt: str,
    model: LLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    batch_size: int = BATCH_SIZE,
) -> str:
    sampling_params = SamplingParams(temperature=0.8)
    results = []
    for i in range(0, len(df), batch_size):
        batch_inputs = df[i:i + batch_size]
        batch_prompts = [
            prepare_message(row["input"], system_prompt, tokenizer)
            for _, row in batch_inputs.iterrows()
        ]
        outputs = model.generate(batch_prompts, sampling_params)
        results.extend([{**row, model_name: output} for row, output in zip(batch_inputs, outputs)])
    return results


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
    