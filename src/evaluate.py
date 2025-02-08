import os
import json
from openai import AsyncOpenAI


from util import load_prompt
from api_client import openai_completion


def prepare_prompt(
    prompt_file: str,
    input_text: str,
    output_text: str,
    pred: str,
    eval_aspect: str,
) -> float:
    """Prepare prompt for GPT-4 evaluation.
    Args:
        prompt_file (str): Prompt file name.
        input_text (str): Input text.
        output_text (str): Output text.
        pred (str): Predicted text.
        eval_aspect (str): Evaluation aspect.
    Returns:
        str: Prepared prompt.
    """
    prompt = load_prompt(prompt_file)
    prompt = prompt.format(
        input_text=input_text,
        output_text=output_text,
        pred=pred,
        eval_aspect=eval_aspect,
    )
    return prompt


async def gpt4eval(
    pred: str, input_text: str, output_text: str, eval_aspect: str, judge_model: str
) -> int | None:
    """Evaluate the output of a model using OpenAI API.
    Args:
        pred (str): Predicted text.
        input_text (str): Input text.
        output_text (str): Output text.
        eval_aspect (str): Evaluation aspect.
        judge_model (str): Judge model name.
    Returns:
        int | None: GPT-4 score or None if an error occurs.
    """
    prompt = prepare_prompt(
        prompt_file="v2.txt",
        input_text=input_text,
        output_text=output_text,
        pred=pred,
        eval_aspect=eval_aspect,
    )
    response = await openai_completion(
        client=client,
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
        response_format={"type": "json_object"},
    )
    try:
        gpt4score = json.loads(response.choices[0].message.content)["score"]
        gpt4score = float(gpt4score)
    except Exception as e:
        print(f"Error occurred: {e}")
        gpt4score = None
    return gpt4score
