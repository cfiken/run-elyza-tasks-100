from enum import Enum
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import Dataset


class APIBASED_MODEL_PROVIDERS(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    CLAUDE = "claude"


def load_model_and_tokenizer(model_name: str, fast: bool, tp: int) -> tuple[AutoModelForCausalLM | LLM, AutoTokenizer]:
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
            tensor_parallel_size=tp,
            dtype="float32",
            gpu_memory_utilization=0.9
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
    dataset: Dataset,
    system_prompt: str,
    model_name: str,
    fast: bool,
    batch_size: int,
    tp: int,
) -> str:
    model, tokenizer = load_model_and_tokenizer(model_name, fast, tp)
    if fast:
        results = _predict_vllm(dataset, system_prompt, model, model_name, tokenizer, batch_size)
    else:
        results = _predict_huggingface(dataset, system_prompt, model, model_name, tokenizer)
    return results


def _predict_huggingface(
    dataset: Dataset,
    system_prompt: str,
    model: AutoModelForCausalLM,
    model_name: str,
    tokenizer: AutoTokenizer,
) -> str:
    results = []
    for row in dataset:
        prompt = prepare_message(row["input"], system_prompt, tokenizer)
        token_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=2048,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(
            output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True
        )
        results.append({**row, model_name: output})
    return results


def _predict_vllm(
    dataset: Dataset,
    system_prompt: str,
    model: LLM,
    model_name: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
) -> str:
    sampling_params = SamplingParams(temperature=1.0, max_tokens=2048)
    results = []
    for i in range(0, len(dataset), batch_size):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_data = dataset.select(batch_indices)
        batch_prompts = [
            prepare_message(row["input"], system_prompt, tokenizer)
            for row in batch_data
        ]
        outputs = model.generate(batch_prompts, sampling_params)
        for j, output in enumerate(outputs):
            results.append({
                **batch_data[j],
                model_name: output.outputs[0].text
            })
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
    