from pathlib import Path
import logging
import time
import pandas as pd

from datasets import load_dataset

from predict import predict
from evaluate import gpt4eval
from util import get_logger

OUTPUT_DIR = Path("output")
LOGGER = get_logger(__name__)
SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"


def inference(model_name: str, fast: bool) -> list[dict[str, str]]:
    """Run model inference on ELYZA tasks 100 dataset and evaluate the outputs via LLM as a judge."""
    dataset = load_dataset("elyza/ELYZA-tasks-100", revision="1.0.0")
    LOGGER.info("Start inference.")
    start = time.time()
    results = predict(dataset, SYSTEM_PROMPT, model_name, fast)
    LOGGER.info(f"Inference complete. Time: {time.time() - start / 60:.2f} minutes")
    return results


async def evaluate(df: pd.DataFrame, model_name: str, judge_model: str) -> list[float]:
    """Evaluate the model."""
    LOGGER.info("Start evaluation.")
    start = time.time()
    scores = []
    for _, row in df.iterrows():
        input_text = row["input"]
        output_text = row["output"]
        eval_aspect = row["eval_aspect"]
        pred = row[model_name]
        score = await gpt4eval(pred, input_text, output_text, eval_aspect, judge_model)
        scores.append(score)
    LOGGER.info(f"Evaluation complete. Time: {time.time() - start / 60:.2f} minutes")
    return scores


async def run(model_name: str, judge_model: str, fast: bool):
    """Run ELYZA tasks 100."""
    predictions = inference(model_name, fast)
    df = pd.DataFrame(predictions)
    scores = await evaluate(df, model_name, judge_model)
    df["score"] = scores
    path = OUTPUT_DIR / f"{model_name.split('/')[-1]}_{judge_model}.csv"
    df.to_csv(path, index=False)
    LOGGER.info(f"Results have been saved successfully. Path: {path}")

async def main(model_name: str, judge_model: str, fast: bool):
    """Main function to run ELYZA tasks 100."""
    try:
        await run(model_name, judge_model, fast)
        LOGGER.info("Results have been saved successfully.")
    except Exception as e:
        LOGGER.error(f"Error occurred: {e}")
        raise e


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--judge_model", "-j", type=str, default="gpt-4o")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    import asyncio
    asyncio.run(main(args.model, args.judge_model, args.fast))
