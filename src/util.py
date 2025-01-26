import json


def load_prompt(file_name: str) -> str:
    with open(f"prompt/{file_name}", "r") as f:
        return f.read()


def save_json(data: dict, file_name: str) -> None:
    with open(f"data/{file_name}", "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
