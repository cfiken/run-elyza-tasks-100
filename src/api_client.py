import os
from openai import AsyncOpenAI

from tenacity import (
  retry,
  stop_after_attempt,
  wait_random_exponential,
)


# https://beta.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def _openai_completion(client: AsyncOpenAI, **kwargs) -> dict:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return await client.chat.completions.create(**kwargs)


async def openai_completion(
    model_name: str,
    user_prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
    response_format: dict | None = None,
) -> dict:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return await _openai_completion(
        client=client,
        model=model_name,
        messages=messages,
        temperature=temperature,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        response_format=response_format,
    )


async def openai_evaluate(
    judge_model: str,
    user_prompt: str,
    system_prompt: str | None = None,
) -> dict:
    return await openai_completion(
        model_name=judge_model,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"},
    )


async def genimi_completion(model_name: str, user_prompt: str, system_prompt: str | None = None) -> dict:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GENIMI_API_KEY"))
    except ImportError:
        raise ImportError("google-generativeai is not installed. Please install it with `pip install google-generativeai`.")

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt
    )
    generation_config = genai.GenerationConfig(
        temperature=0.0,
    )
    response = model.generate_content(
        user_prompt,
        generation_config=generation_config,
    )
    return response.text


async def bedrock_completion(model_name: str, user_propmpt: str, system_prompt: str | None = None) -> dict:
    try:
        import boto3
    except ImportError as e:
        raise ImportError("boto3 is not installed. Please install it with `pip install boto3`.")
    client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_propmpt})
    model_response = client.converse(
        modelId=model_name,
        messages=messages
    )
    return model_response