import json
import logging

def load_prompt(file_name: str) -> str:
    with open(f"prompt/{file_name}", "r") as f:
        return f.read()


def save_json(data: dict, file_name: str) -> None:
    with open(f"data/{file_name}", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
