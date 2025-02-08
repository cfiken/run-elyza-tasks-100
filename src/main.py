from pathlib import Path
import logging
import pandas as pd

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from predict import predict
from evaluate import gpt4eval

OUTPUT_DIR = Path("output")
LOGGER = logging.getLogger(__name__)
SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"


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


def inference(model_name: str) -> list[dict[str, str]]:
    """Run model inference on ELYZA tasks 100 dataset and evaluate the outputs via LLM as a judge."""
    dataset = load_dataset("elyza/ELYZA-tasks-100", revision="1.0.0")
    model, tokenizer = load_model_and_tokenizer(model_name)
    results = []
    for example in dataset["test"]:
        prediction = predict(example["input"], SYSTEM_PROMPT, model, tokenizer)
        results.append({**example, model_name: prediction})
    LOGGER.info("Processing complete")
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
        LOGGER.info("Results have been saved successfully.")
    except Exception as e:
        LOGGER.error(f"Error occurred: {e}")
        raise e


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--judge_model", "-j", type=str, default="gpt-4o")
    args = parser.parse_args()

    import asyncio
    asyncio.run(main(args.model, args.judge_model))
