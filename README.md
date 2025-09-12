# LLM-Driven Text Classification Experiments

This project applies Large Language Models (LLMs) to classify short texts into predefined categories and compares three prompting strategies.

## Methods
- **Direct category selection (Zero-shot prompting)**  
  The model selects one category from the full list without additional hints.

- **Category selection with keywords (Few-shot prompting)**  
  Each category is enriched with keywords aggregated from ads in that category; the model picks the single best category.

- **Top-3 selection + pairwise comparison (Prompt Chaining)**  
  The model proposes three most likely categories from the keyword-augmented list, then a pairwise evaluator ranks these candidates using precomputed inter-category differences derived from ad texts.

## Project Layout
```
src/
static/
.env
requirements.txt
Dockerfile
Makefile
```

## Prerequisites
- Docker and Make installed
- API keys in `.env`

Example `.env` snippet:
```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
# add any other keys your client code expects
```

## Dataset
- Place the dataset into the `static/` directory.  
  *https://github.com/dimazharkov/datasets/tree/main/ads-board/data*

## Build Image & Create Container
Build the image and create a long-running container (environment variables loaded from `.env`):
```bash
make create
```

## Running Experiments
Run experiments inside the already running container:
```bash
make exec-one    # python -m src.main experiment one
make exec-two    # python -m src.main experiment two
make exec-three  # python -m src.main experiment three
```

## Outputs
Results are written to the `static/` directory:
- `experiment_one.json`
- `experiment_two.json`
- `experiment_three.json`

## Maintenance
```bash
make logs    # follow logs
make shell   # open interactive shell inside the container
make stop    # stop the container
make start   # start the container again
```

## Cleanup
```bash
make clean   # stop & remove container and image