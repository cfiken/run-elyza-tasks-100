import os
import json
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tenacity import (
  retry,
  stop_after_attempt,
  wait_random_exponential,
)
from openai import AsyncOpenAI

from util import load_prompt
from prediction import predict


OUTPUT_DIR = Path("output")


def load_model_and_tokenizer(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer.

    Args:
        model_name (str): Model name.
    Returns:
        tuple: Model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    model.eval()
    return model, tokenizer


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


# https://beta.openai.com/docs/guides/rate-limits/retrying-with-exponential-backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def completion(client: AsyncOpenAI, **kwargs):
    return await client.chat.completions.create(**kwargs)


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
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = prepare_prompt(
        prompt_file="v2.txt",
        input_text=input_text,
        output_text=output_text,
        pred=pred,
        eval_aspect=eval_aspect,
    )
    response = await completion(
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


def inference(model_name: str) -> list[dict[str, str]]:
    """Run model inference on ELYZA tasks 100 dataset and evaluate the outputs via LLM as a judge."""
    dataset = load_dataset("elyza/ELYZA-tasks-100", revision="1.0.0")
    model, tokenizer = load_model_and_tokenizer(model_name)
    results = []
    for example in dataset["test"]:
        prediction = predict(example, model, tokenizer)
        results.append({**example, model_name: prediction})
    print("Processing complete")
    return results


async def run(model_name: str, judge_model: str):
    """Run ELYZA tasks 100."""
    predictions = inference(model_name)
    df = pd.DataFrame(predictions)
    scores = []
    for _, row in df.iterrows():
        input_text = row["input"]
        output_text = row["output"]
        eval_aspect = row["eval_aspect"]
        pred = row[model_name]
        score = await gpt4eval(pred, input_text, output_text, eval_aspect, judge_model)
        scores.append(score)
    df["score"] = scores
    path = OUTPUT_DIR / f"{model_name.split('/')[-1]}_{judge_model}.csv"
    df.to_csv(path, index=False)


async def main(model_name: str, judge_model: str):
    """Main function to run ELYZA tasks 100."""
    try:
        await run(model_name, judge_model)
        print("Results have been saved successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--judge_model", "-j", type=str, default="gpt-4o")
    args = parser.parse_args()

    import asyncio
    asyncio.run(main(args.model, args.judge_model))
