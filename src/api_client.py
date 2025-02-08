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


async def openai_completion(judge_model: str, prompt: str) -> dict:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return await _openai_completion(
        client=client,
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"},
    )