# PRICE Evaluation

Official evaluation toolkit for **PRICE: Prediction of Real-world Interactions with Constraints Evaluation**.

PRICE evaluates instruction-conditional image-to-image generation grounded in real-world robot, egocentric, and third-person interaction data. Given an initial-state image and an instruction, a model predicts the resulting scene. VLM judges score instruction following, scene consistency, and physical plausibility.

Benchmark data is hosted separately at [BAAI/PRICE](https://huggingface.co/datasets/BAAI/PRICE). This directory contains the PRICE evaluation code and prompts for the [Orca](https://github.com/orca-wm/Orca) project.

## Repository layout

```text
Orca/evaluation/image_gen/PRICE/
├── eval_prompt/                 # Judge prompt templates
├── envs/gemma4/                 # Optional isolated local-judge environment
├── scripts/
│   ├── download_dataset.py      # Download BAAI/PRICE from Hugging Face
│   ├── price_dataset.py         # Shared dataset loader
│   ├── eval_imgs.py             # Main API/local judge entry point
│   └── eval_imgs_with_gemma4.py # Optional local Gemma 4 judge
├── env.example.toml
└── pyproject.toml
```

Downloaded benchmark files are shared under `Orca/evaluation/data/` and ignored by Git.

## Installation

The evaluator requires Python 3.10 or later.

```bash
git clone https://github.com/orca-wm/Orca.git
cd Orca/evaluation/image_gen/PRICE

python -m venv .venv
source .venv/bin/activate
pip install .
```

You may also use `uv`:

```bash
uv venv
uv pip install .
```

## 1. Download the benchmark

From `Orca/evaluation/image_gen/PRICE`, download the dataset before running evaluation:

```bash
python scripts/download_dataset.py
```

By default this downloads `BAAI/PRICE@main` into `Orca/evaluation/data/PRICE`. The shared downloader provides the same result from `Orca/evaluation`:

```bash
cd ../..
python download_datasets.py price
cd image_gen/PRICE
```

To pin a released version with the PRICE-specific downloader:

```bash
python scripts/download_dataset.py --revision v0.1
```

While the dataset repository is private, authenticate once with a Hugging Face account that has access:

```bash
hf auth login
python scripts/download_dataset.py
```

Authentication is not required after the dataset becomes public.

Useful download options:

```bash
python scripts/download_dataset.py \
  --repo-id BAAI/PRICE \
  --revision main \
  --local-dir ../../data/PRICE
```

If you already have a prepared local copy of the HF dataset, skip the download and pass its path with `--dataset-dir` when evaluating.

## 2. Prepare model predictions

Place generated images under an output directory and name each file by its PRICE sample ID:

```text
OUTPUT_ROOT/
└── imgs/
    ├── agibot_world_1.png
    ├── agibot_world_2.png
    └── ...
```

The evaluator matches each filename against `data/test/metadata.jsonl` in the downloaded HF dataset. Unknown IDs are skipped with a warning.

## 3. Configure API judges

Copy the example configuration:

```bash
cp env.example.toml env.toml
```

Add credentials only for the providers you intend to use:

```toml
[openai]
api_key = "YOUR_OPENAI_API_KEY"

[gemini]
api_key = "YOUR_GEMINI_API_KEY"

[doubao]
api_key = "YOUR_DOUBAO_API_KEY"

[minimax]
api_key = "YOUR_MINIMAX_API_KEY"
```

The supported API judge aliases are:

| Alias | Provider |
| --- | --- |
| `gemini3.1` | Gemini official API or AiHubMix fallback |
| `gpt5.4` | OpenAI official API or AiHubMix fallback |
| `doubao2.0` | Doubao official API or AiHubMix fallback |
| `minimax-m3.1-flash` | MiniMax official API (recommended MiniMax judge) |
| `minimax-m3` | MiniMax official API or existing AiHubMix fallback |

MiniMax judges run in Flash, then M3 order in `--all-api-models` and `--all-models`. Flash requires a valid `[minimax]` key with `--provider-mode auto` or `official`; its aggregator route is not supported. The explicit legacy `--model minimax-m2.7` remains accepted, and existing legacy summaries remain included when regenerating score tables. It is excluded from ordinary judge choices and all-model runs.

The MiniMax provider defaults to the global endpoint (`https://api.minimax.io/v1`). To use the mainland China site instead, set `base_url = "https://api.minimaxi.com/v1"` under `[minimax]` in `env.toml`.

If an official provider key is unavailable, configure the optional fallback:

```toml
[aihubmix]
api_key = "YOUR_AIHUBMIX_API_KEY"
```

Use `--provider-mode auto` (default), `official`, or `aihubmix` to control routing.

## 4. Run evaluation

Run one API judge:

```bash
python scripts/eval_imgs.py OUTPUT_ROOT --model gemini3.1
```

The default dataset path is `Orca/evaluation/data/PRICE`. Override it when needed:

```bash
python scripts/eval_imgs.py OUTPUT_ROOT \
  --dataset-dir /path/to/PRICE-dataset \
  --model gemini3.1
```

Evaluate a subset:

```bash
python scripts/eval_imgs.py OUTPUT_ROOT \
  --ids agibot_world_1,agibot_world_2
```

Run all API judges and generate a combined score table:

```bash
python scripts/eval_imgs.py OUTPUT_ROOT --all-api-models
```

Run the local Gemma 4 judge plus all API judges:

```bash
python scripts/eval_imgs.py OUTPUT_ROOT --all-models
```

Regenerate the combined table from existing summary files:

```bash
python scripts/eval_imgs.py OUTPUT_ROOT --write-judge-scores-md
```

### Outputs

Each judge writes:

```text
OUTPUT_ROOT/eval_results.<prompt>.<model>.json
OUTPUT_ROOT/eval_results_summary.<prompt>.<model>.json
```

Multi-judge runs additionally write:

```text
OUTPUT_ROOT/scores_of_judges.md
```

The default judge prompt is `eval_prompt/0601.py`.

## Local Gemma 4 judge

The optional Gemma 4 judge uses an isolated CUDA environment so the heavyweight `torch` and `transformers` dependencies do not affect the API-only installation.

```bash
cd envs/gemma4
uv sync
cp env.example.toml env.toml
# edit env.toml, then return to evaluation/image_gen/PRICE/
cd ../..
```

Set the local checkpoint path in `env.toml`, return to the `evaluation/image_gen/PRICE/` directory, and run:

```bash
python scripts/eval_imgs.py OUTPUT_ROOT --model gemma4
```

See [envs/gemma4/README.md](envs/gemma4/README.md) for hardware requirements and advanced options.

## Dataset format

`BAAI/PRICE` uses Hugging Face's multi-image ImageFolder layout. Each metadata row contains:

- `id`: stable sample identifier;
- `query`: initial-state image;
- `output`: reference target-state image;
- `lang`: action instruction;
- `dataset`: source collection;
- provenance fields, timestamps, indices, seed, and checksums.

The evaluator uses `query` and `lang`; reference targets are never sent to the judge.

## Tests

Run the offline dataset and API request regression tests with:

```bash
python -m unittest discover -s tests
```

## Citation

PRICE-V0.1 is introduced as part of the *Orca* technical report by the Orca Team at the Beijing Academy of Artificial Intelligence. Full citation details will be added when the report is released.

## License

The PRICE evaluation code is released under the [Apache License 2.0](LICENSE).
