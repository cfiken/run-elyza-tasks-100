# Run ELYZA Tasks 100

This repository contains the code to run the ELYZA Tasks 100.

# Usage

To run the ELYZA Tasks 100, you can use the following command:

## 1. Setup environment

Setup `.env`.

```sh
cp .env.example .env
vi .env # set OPENAI_API_KEY
```

Build docker image and create the output directory for saving the results.

```sh
$ docker compose build
$ mkdir output && chmod 777 output
```

## 2. Run the ELYZA Tasks 100

```sh
$ docker compose run --rm app poetry run python src/main.py --model abeja/ABEJA-Qwen2.5-32b-Japanese-v0.1 --judge_model gpt-4o
```

If you want to run with the api-based model, you can use the following command:

```sh
$ docker compose run --rm app poetry run python src/main.py --model openai/gpt-4o --judge_model gpt-4o --api
```

If you want to run with the local model, use `-v` to mount the model directory or change docker-compose.yml:

```sh
$ docker compose run -v /local/model/path:/app/model_dir_name --rm app poetry run python src/main.py --model ./model_dir_name --judge_model gpt-4o
```

Then, you can find the results in `output` directory.

## Other ways (dockerhub image)

Run with dockerhub image without repository code.

```sh
$ mkdir output && chmod 777 output
$ docker run -v $(pwd)/output:/app/output cfiken/elyza-tasks-100:latest poetry run python src/main.py --model abeja/ABEJA-Qwen2.5-32b-Japanese-v0.1 --judge_model gpt-4o
```

Then, you can find the results in current or mounted directory.

# TODO

- [ ] Add api-based model support
- [ ] Add custom prompt support
- [ ] Faster prediction via batch or library
