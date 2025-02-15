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


async def openai_completion(judge_model: str, user_prompt: str, system_prompt: str | None = None) -> dict:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return await _openai_completion(
        client=client,
        model=judge_model,
        messages=messages,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"},
    )


async def genimi_completion(model: str, user_prompt: str, system_prompt: str | None = None) -> dict:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GENIMI_API_KEY"))
    except ImportError:
        raise ImportError("google-generativeai is not installed. Please install it with `pip install google-generativeai`.")

    model = genai.GenerativeModel(
        model_name=model,
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