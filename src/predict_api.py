from typing import Callable
from enum import Enum
import asyncio
from datasets import Dataset


class APIBASED_MODEL_PROVIDERS(Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    GEMINI = "gemini"


async def predict_api(
    dataset: Dataset,
    system_prompt: str,
    model_name: str,
    batch_size: int,
) -> str:
    results = await _predict_api(dataset, system_prompt, model_name, batch_size)
    return results


def _load_completion_func(model_type: str) -> Callable:
    if model_type == "openai":
        from api_client import openai_completion
        return openai_completion
    elif model_type == "bedrock":
        from api_client import bedrock_completion
        return bedrock_completion
    elif model_type == "gemini":
        from api_client import genimi_completion
        return genimi_completion


def _parse_model_output(model_type: str, output: dict) -> str:
    if model_type == "openai":
        return output.choices[0].message.content
    elif model_type == "bedrock":
        return output["output"]["message"]["content"][0]["text"]
    elif model_type == "gemini":
        return output.text


async def _predict_api(
    dataset: Dataset,
    system_prompt: str,
    model: str,
    batch_size: int,
) -> str:
    model_type, model_name = model.split("/")
    completion_func = _load_completion_func(model_type)
    results = []
    for i in range(0, len(dataset), batch_size):
        batch_indices = range(i, min(i + batch_size, len(dataset)))
        batch_data = dataset.select(batch_indices)
        tasks = [
            completion_func(
                model_name=model_name,
                user_prompt=prompt,
                system_prompt=system_prompt,
            )
            for prompt in batch_data["input"]
        ]
        outputs = await asyncio.gather(*tasks)
        for j, output in enumerate(outputs):
            results.append({
                **batch_data[j],
                model: _parse_model_output(model_type, output)
            })
    return results


def prepare_message(user_prompt: str, system_prompt: str) -> dict[str, str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages
    