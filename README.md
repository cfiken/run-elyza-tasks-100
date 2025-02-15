# Run ELYZA Tasks 100

This repository contains the code to run the ELYZA Tasks 100.

# Usage

To run the ELYZA Tasks 100, you can use the following command:

## 1. Use repository code

Setup `.env`.

```sh
cp .env.example .env
vi .env # set OPENAI_API_KEY
```

Build docker image. 
Create the output directory.
Run with volume mount.

```sh
$ docker compose build
$ mkdir output && chmod 777 output
$ docker compose run --rm app poetry run python src/main.py --model abeja/ABEJA-Qwen2.5-32b-Japanese-v0.1 --judge_model gpt-4o
```

Then, you can find the results in `output` directory.

## 2. Use dockerhub image

Run with dockerhub image.

```sh
$ mkdir output && chmod 777 output
$ docker run -v $(pwd)/output:/app/output cfiken/elyza-tasks-100:latest poetry run python src/main.py --model abeja/ABEJA-Qwen2.5-32b-Japanese-v0.1 --judge_model gpt-4o
```

Then, you can find the results in current or mounted directory.

# TODO

- [ ] Add api-based model support
- [ ] Add custom prompt support
- [ ] Faster prediction via batch or library
