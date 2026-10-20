#!/usr/bin/env python3
"""Evaluate PRICE predictions with API or local VLM judges.

Benchmark metadata and initial-state images are loaded from a local copy of
the BAAI/PRICE Hugging Face dataset. Download it with download_dataset.py.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import price_dataset

API_BASE_URL = "https://api.inferera.com/v1"
MAX_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2

API_JUDGE_MODEL_ALIASES = {
    "gemini3.1": "gemini-3.1-pro-preview",
    "gpt5.4": "gpt-5.4-high",
    "doubao2.0": "doubao-seed-2-0-pro",
    "minimax-m3.1-flash": "MiniMax-M3.1-Flash",
    "minimax-m3": "MiniMax-M3",
}

# Keep explicit legacy requests and historical score files addressable.
LEGACY_API_JUDGE_MODEL_ALIASES = {"minimax-m2.7": "MiniMax-M2.7"}

# Optional direct-to-provider routing (see docs/PUBLIC_OFFICIAL_API_JUDGES_PLAN_zh.md).
# [aihubmix] in env.toml remains the default fallback; configuring a provider's own
# table (below) routes that judge alias straight to the provider's official API instead.
JUDGE_ALIAS_PROVIDER = {
    "gemini3.1": "gemini",
    "gpt5.4": "openai",
    "doubao2.0": "doubao",
    "minimax-m3.1-flash": "minimax",
    "minimax-m3": "minimax",
    "minimax-m2.7": "minimax",
}

# NOTE: these base URLs are the providers' publicly documented endpoints, but the
# request/response compatibility with our OpenAI-style client has not been verified
# against a real key yet (see the plan doc's "实现状态" section).
PROVIDER_DEFAULT_BASE_URL = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    # Global (international) OpenAI-compatible endpoint. For the mainland China
    # site set base_url = "https://api.minimaxi.com/v1" under [minimax] in env.toml.
    "minimax": "https://api.minimax.io/v1",
}

# NOTE: unverified best-effort defaults, reusing the aggregator's alias strings (a
# passthrough aggregator commonly proxies the same official model id). Confirm and
# correct these once a real key is available for each provider.
JUDGE_ALIAS_OFFICIAL_MODEL_ID = {
    "gemini3.1": "gemini-3.1-pro-preview",
    "gpt5.4": "gpt-5.4-high",
    "doubao2.0": "doubao-seed-2-0-pro",
    "minimax-m3.1-flash": "MiniMax-M3.1-Flash",
    "minimax-m3": "MiniMax-M3",
    "minimax-m2.7": "MiniMax-M2.7",
}

PROVIDER_MODE_CHOICES = ("auto", "official", "aihubmix")

GEMMA4_JUDGE_ALIAS = "gemma4"
LOCAL_JUDGE_ALIASES = frozenset({GEMMA4_JUDGE_ALIAS})
# Preserve the existing order of other providers; recommend Flash first for MiniMax.
ACTIVE_API_JUDGE_ALIASES = sorted(
    alias for alias in API_JUDGE_MODEL_ALIASES if JUDGE_ALIAS_PROVIDER[alias] != "minimax"
) + ["minimax-m3.1-flash", "minimax-m3"]
ALL_JUDGE_ALIASES = ACTIVE_API_JUDGE_ALIASES + sorted(LOCAL_JUDGE_ALIASES)
JUDGE_SCORES_MD_ALIAS_ORDER = [
    GEMMA4_JUDGE_ALIAS, *ACTIVE_API_JUDGE_ALIASES, *LEGACY_API_JUDGE_MODEL_ALIASES
]

DEFAULT_JUDGE_MODEL = "gemini3.1"
DEFAULT_EVAL_PROMPT = "0601.py"
GEMMA4_SCRIPT_NAME = "eval_imgs_with_gemma4.py"

MODEL = ""
MIN_SCORE = 0
MAX_SCORE = 0
_eval_prompt_template = ""


@dataclass(frozen=True)
class EvalPromptConfig:
    template: str
    min_score: int
    max_score: int


def _public_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_git_commit_hash() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_public_root(),
        )
        commit = result.stdout.strip()
        return commit or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_env_config(path: Path | None = None) -> dict[str, Any]:
    env_path = path or (_public_root() / "env.toml")
    if not env_path.is_file():
        raise FileNotFoundError(
            f"env config not found: {env_path}. Copy env.example.toml to env.toml "
            "and fill in your AiHubMix API key."
        )

    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TOML support is unavailable. Use Python 3.11+ or install dependencies "
                "with: pip install ."
            ) from exc

    with env_path.open("rb") as f:
        config = tomllib.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"env config must be a TOML table: {env_path}")
    return config


def resolve_aihubmix_api_key(config: dict[str, Any]) -> str:
    aihubmix = config.get("aihubmix")
    if not isinstance(aihubmix, dict):
        raise ValueError("env.toml must contain an [aihubmix] table")
    api_key = str(aihubmix.get("api_key", "")).strip()
    if not api_key or api_key == "YOUR_AIHUBMIX_API_KEY":
        raise ValueError("env.toml must set aihubmix.api_key")
    return api_key


def _provider_table(provider: str, env_config: dict[str, Any]) -> dict[str, Any] | None:
    table = env_config.get(provider)
    return table if isinstance(table, dict) else None


def has_valid_official_key(provider: str, env_config: dict[str, Any]) -> bool:
    table = _provider_table(provider, env_config)
    if table is None:
        return False
    api_key = str(table.get("api_key", "")).strip()
    return bool(api_key) and not api_key.startswith("YOUR_")


def resolve_alias_route(
    alias: str,
    env_config: dict[str, Any],
    *,
    provider_mode: str = "auto",
) -> tuple[str, str]:
    """Return (route, provider): route is "official" or "aihubmix"."""
    if provider_mode not in PROVIDER_MODE_CHOICES:
        raise ValueError(f"invalid provider_mode: {provider_mode!r}")

    provider = JUDGE_ALIAS_PROVIDER[alias]

    # Flash's aggregator model ID and availability have not been verified.
    if alias == "minimax-m3.1-flash" and (
        provider_mode == "aihubmix" or not has_valid_official_key(provider, env_config)
    ):
        raise ValueError(
            "minimax-m3.1-flash requires a valid [minimax] api_key and auto or "
            "official routing; its aihubmix route is not supported"
        )

    if provider_mode == "official":
        if not has_valid_official_key(provider, env_config):
            raise ValueError(
                f"--provider-mode official requires a valid [{provider}] table "
                f"(with a real api_key) in env.toml for judge alias {alias!r}"
            )
        return "official", provider

    if provider_mode == "aihubmix":
        return "aihubmix", provider

    # auto: prefer the official provider if a real key is configured, else fall
    # back to the shared aihubmix aggregator.
    if has_valid_official_key(provider, env_config):
        return "official", provider
    return "aihubmix", provider


def resolve_official_model_value(provider: str, alias: str, env_config: dict[str, Any]) -> str:
    if provider == "doubao":
        table = _provider_table(provider, env_config) or {}
        endpoint_id = str(table.get("endpoint_id", "")).strip()
        if endpoint_id:
            return endpoint_id
    return JUDGE_ALIAS_OFFICIAL_MODEL_ID[alias]


def resolve_model_value(route: str, alias: str, provider: str, env_config: dict[str, Any]) -> str:
    if route == "aihubmix":
        return resolve_judge_model_id(alias)
    return resolve_official_model_value(provider, alias, env_config)


def build_client(route: str, provider: str, env_config: dict[str, Any]) -> Any:
    import openai

    if route == "aihubmix":
        api_key = resolve_aihubmix_api_key(env_config)
        return openai.OpenAI(api_key=api_key, base_url=API_BASE_URL)

    table = _provider_table(provider, env_config)
    if table is None or not has_valid_official_key(provider, env_config):
        raise ValueError(f"env.toml must set a valid api_key under [{provider}]")
    api_key = str(table["api_key"]).strip()
    base_url = str(table.get("base_url") or PROVIDER_DEFAULT_BASE_URL[provider])
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def add_ids_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated sample ids to evaluate.",
    )
    parser.add_argument(
        "--ids-file",
        type=str,
        default=None,
        help="Path to a file with one sample id per line (# comments allowed).",
    )


def resolve_sample_ids(
    ids: str | None = None,
    *,
    ids_file: str | None = None,
) -> set[str] | None:
    result: set[str] = set()
    if ids:
        for part in ids.split(","):
            sample_id = part.strip()
            if sample_id:
                result.add(sample_id)
    if ids_file:
        path = Path(ids_file).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ids file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                sample_id = line.strip()
                if not sample_id or sample_id.startswith("#"):
                    continue
                result.add(sample_id)
    return result if result else None


def resolve_sample_ids_from_args(args: argparse.Namespace) -> set[str] | None:
    return resolve_sample_ids(args.ids, ids_file=args.ids_file)


def warn_missing_sample_ids(
    requested: set[str],
    found: set[str],
    *,
    context: str,
) -> None:
    missing = requested - found
    for sample_id in sorted(missing):
        print(f"[WARN] requested id not found in {context}: {sample_id}")


def resolve_judge_model_id(alias: str) -> str:
    if alias in LEGACY_API_JUDGE_MODEL_ALIASES:
        return LEGACY_API_JUDGE_MODEL_ALIASES[alias]
    return API_JUDGE_MODEL_ALIASES[alias]


def parse_judge_alias(alias: str) -> str:
    if alias not in ALL_JUDGE_ALIASES and alias not in LEGACY_API_JUDGE_MODEL_ALIASES:
        raise argparse.ArgumentTypeError(f"unknown judge alias: {alias!r}")
    return alias


def eval_prompt_template_name(eval_prompt: str | Path) -> str:
    name = Path(eval_prompt).expanduser().stem.strip()
    if not name:
        raise ValueError(f"invalid eval prompt filename: {eval_prompt}")
    return name


def eval_results_path(output_root: Path, *, template: str, model: str) -> Path:
    return output_root / f"eval_results.{template}.{model}.json"


def eval_results_summary_path(output_root: Path, *, template: str, model: str) -> Path:
    return output_root / f"eval_results_summary.{template}.{model}.json"


def resolve_eval_prompt_path(eval_prompt: str | Path | None) -> Path:
    if eval_prompt is not None:
        return Path(eval_prompt).expanduser().resolve()
    return (_public_root() / "eval_prompt" / DEFAULT_EVAL_PROMPT).resolve()


def load_eval_prompt_config(path: Path) -> EvalPromptConfig:
    prompt_path = path.expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"eval prompt file not found: {prompt_path}")

    spec = importlib.util.spec_from_file_location("eval_prompt_module", prompt_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load eval prompt module: {prompt_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    template = getattr(module, "EVAL_PROMPT_TEMPLATE", None)
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"EVAL_PROMPT_TEMPLATE must be a non-empty string: {prompt_path}")

    min_score = getattr(module, "MIN_SCORE", None)
    max_score = getattr(module, "MAX_SCORE", None)
    if isinstance(min_score, bool) or not isinstance(min_score, int):
        raise ValueError(f"MIN_SCORE must be an integer: {prompt_path}")
    if isinstance(max_score, bool) or not isinstance(max_score, int):
        raise ValueError(f"MAX_SCORE must be an integer: {prompt_path}")
    if min_score > max_score:
        raise ValueError(
            f"MIN_SCORE must be <= MAX_SCORE ({min_score} > {max_score}): {prompt_path}"
        )

    return EvalPromptConfig(
        template=template,
        min_score=min_score,
        max_score=max_score,
    )


def score_to_percentage(score: float, *, min_score: int, max_score: int) -> float:
    if min_score == max_score:
        return 100.0 if score >= max_score else 0.0
    return round((score - min_score) / (max_score - min_score) * 100, 4)


def _apply_eval_prompt_config(config: EvalPromptConfig) -> None:
    global MIN_SCORE, MAX_SCORE, _eval_prompt_template
    MIN_SCORE = config.min_score
    MAX_SCORE = config.max_score
    _eval_prompt_template = config.template


def load_samples(
    output_root: Path,
    *,
    dataset_root: Path,
    ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    imgs_dir = output_root / "imgs"
    if not imgs_dir.is_dir():
        raise FileNotFoundError(f"imgs directory not found: {imgs_dir}")

    manifest = price_dataset.load_manifest(dataset_root)

    samples: list[dict[str, Any]] = []
    found_ids: set[str] = set()
    for output_path in sorted(imgs_dir.glob("*.png")):
        sample_id = output_path.stem
        if ids is not None and sample_id not in ids:
            continue

        entry = manifest.get(sample_id)
        if entry is None:
            print(f"[WARN] {output_path} does not match a benchmark sample id, skip")
            continue

        instruction = str(entry.get("lang", "")).strip()
        query_path = str(entry.get("query_path", "")).strip()
        if not instruction or not query_path:
            print(f"[WARN] incomplete manifest entry for id={sample_id}, skip")
            continue

        input_path = Path(query_path)
        if not input_path.is_file():
            print(f"[WARN] input image missing for id={sample_id}: {input_path}, skip")
            continue

        samples.append(
            {
                "id": sample_id,
                "instruction": instruction,
                "input_image": str(input_path.resolve()),
                "output_image": str(output_path.resolve()),
            }
        )
        found_ids.add(sample_id)

    if ids is not None:
        warn_missing_sample_ids(ids, found_ids, context=str(output_root))

    samples.sort(key=lambda item: item["id"])
    if not samples:
        raise RuntimeError(f"no valid samples found under {output_root}")
    return samples


def encode_image_base64(image_path: str | Path) -> str:
    from PIL import Image

    image = Image.open(image_path)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _image_content(image_path: str | Path) -> dict[str, Any]:
    b64 = encode_image_base64(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
    }


def build_eval_prompt(instruction: str) -> str:
    return _eval_prompt_template.format(instruction=instruction)


def build_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_eval_prompt(sample["instruction"]),
        },
        _image_content(sample["input_image"]),
        _image_content(sample["output_image"]),
    ]
    return [{"role": "user", "content": content}]


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return stripped


def _normalize_score(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"score must be an integer from {MIN_SCORE} to {MAX_SCORE}, got {value!r}")
    if isinstance(value, (int, float)):
        score = int(value)
        if MIN_SCORE <= score <= MAX_SCORE:
            return score
    raise ValueError(f"score must be an integer from {MIN_SCORE} to {MAX_SCORE}, got {value!r}")


def _score_to_percentage(score: float) -> float:
    return score_to_percentage(score, min_score=MIN_SCORE, max_score=MAX_SCORE)


def _average_metric(
    successful: list[dict[str, Any]], key: str
) -> tuple[Optional[float], Optional[float]]:
    if not successful:
        return None, None
    values = [item[key] for item in successful if key in item]
    if not values:
        return None, None
    average = round(sum(values) / len(values), 4)
    return average, _score_to_percentage(average)


def dimension_average_fields(successful: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    average_score, average_score_percentage = _average_metric(successful, "score")
    fields["average_score"] = average_score
    fields["average_score_percentage"] = average_score_percentage

    if any("instruction_score" in item for item in successful):
        average_instruction_score, average_instruction_score_percentage = _average_metric(
            successful, "instruction_score"
        )
        fields["average_instruction_score"] = average_instruction_score
        fields["average_instruction_score_percentage"] = average_instruction_score_percentage

    if any("physical_score" in item for item in successful):
        average_physical_score, average_physical_score_percentage = _average_metric(
            successful, "physical_score"
        )
        fields["average_physical_score"] = average_physical_score
        fields["average_physical_score_percentage"] = average_physical_score_percentage

    return fields


def parse_eval_response(text: str) -> dict[str, Any]:
    payload = json.loads(_extract_json_text(text))
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    if "reasoning" not in payload:
        raise ValueError("missing reasoning")
    reasoning = str(payload["reasoning"]).strip()
    if not reasoning:
        raise ValueError("empty reasoning")

    if "instruction_score" in payload and "physical_score" in payload:
        instruction_score = _normalize_score(payload["instruction_score"])
        physical_score = _normalize_score(payload["physical_score"])
        score = round((instruction_score + physical_score) / 2.0, 4)
        return {
            "instruction_score": instruction_score,
            "physical_score": physical_score,
            "score": score,
            "reasoning": reasoning,
        }

    if "score" in payload:
        return {
            "score": _normalize_score(payload["score"]),
            "reasoning": reasoning,
        }

    raise ValueError("missing score fields: expected score or instruction_score and physical_score")


def eval_sample(client: Any, sample: dict[str, Any]) -> dict[str, Any]:
    attempt_errors: list[str] = []
    messages = build_messages(sample)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("empty model response")
            parsed = parse_eval_response(content)
            result: dict[str, Any] = {
                "id": sample["id"],
                "instruction": sample["instruction"],
                "input_image": sample["input_image"],
                "output_image": sample["output_image"],
                "status": "success",
                "score": parsed["score"],
                "reasoning": parsed["reasoning"],
            }
            if "instruction_score" in parsed:
                result["instruction_score"] = parsed["instruction_score"]
            if "physical_score" in parsed:
                result["physical_score"] = parsed["physical_score"]
            return result
        except Exception as exc:
            attempt_errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)

    return {
        "id": sample["id"],
        "instruction": sample["instruction"],
        "input_image": sample["input_image"],
        "output_image": sample["output_image"],
        "status": "failed",
        "error": f"Failed after {MAX_ATTEMPTS} attempts: {attempt_errors[-1]}",
        "attempt_errors": attempt_errors,
    }


def compute_summary(
    results: list[dict[str, Any]],
    *,
    output_root: Path,
    concurrency: int,
    eval_prompt_path: Path,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    successful = [item for item in results if item.get("status") == "success"]
    failed = [item for item in results if item.get("status") != "success"]

    return {
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "output_root": str(output_root),
        "dataset_root": str(dataset_root) if dataset_root is not None else None,
        "concurrency": concurrency,
        "total_samples": len(results),
        "successful_samples": len(successful),
        "failed_samples": len(failed),
        "failed_sample_ids": [item["id"] for item in failed],
        "min_score": MIN_SCORE,
        "max_score": MAX_SCORE,
        **dimension_average_fields(successful),
        "git_commit_hash": get_git_commit_hash(),
        "eval_prompt_path": str(eval_prompt_path),
        "eval_prompt_template": _eval_prompt_template,
    }


def run_evaluation(
    output_root: Path,
    *,
    concurrency: int,
    client: Any,
    eval_prompt_path: Path,
    dataset_root: Path,
    ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from tqdm import tqdm

    samples = load_samples(output_root, dataset_root=dataset_root, ids=ids)
    results: list[Optional[dict[str, Any]]] = [None] * len(samples)
    ok_count = 0
    fail_count = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {
            executor.submit(eval_sample, client, sample): idx
            for idx, sample in enumerate(samples)
        }
        progress = tqdm(total=len(samples), desc="Evaluating")
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            result = future.result()
            results[idx] = result
            with lock:
                if result.get("status") == "success":
                    ok_count += 1
                else:
                    fail_count += 1
                progress.set_postfix(ok=ok_count, fail=fail_count, refresh=False)
                progress.update(1)
        progress.close()

    ordered_results = [item for item in results if item is not None]
    summary = compute_summary(
        ordered_results,
        output_root=output_root,
        concurrency=concurrency,
        eval_prompt_path=eval_prompt_path,
        dataset_root=dataset_root,
    )
    return ordered_results, summary


def scores_of_judges_md_path(output_root: Path) -> Path:
    return output_root / "scores_of_judges.md"


def load_judge_summary(
    output_root: Path,
    *,
    template: str,
    alias: str,
) -> dict[str, Any] | None:
    summary_path = eval_results_summary_path(output_root, template=template, model=alias)
    if not summary_path.is_file():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def write_scores_of_judges_md(
    output_root: Path,
    *,
    template: str,
    judge_aliases: list[str] | None = None,
) -> Path:
    aliases = judge_aliases or JUDGE_SCORES_MD_ALIAS_ORDER
    scores: dict[str, float | None] = {}
    for alias in aliases:
        summary = load_judge_summary(output_root, template=template, alias=alias)
        percentage = summary.get("average_score_percentage") if summary else None
        if isinstance(percentage, (int, float)):
            scores[alias] = float(percentage)
        else:
            scores[alias] = None

    present = [alias for alias in aliases if scores[alias] is not None]
    if not present:
        raise RuntimeError(
            f"no judge summary files found under {output_root} for prompt {template}"
        )

    average = round(sum(float(scores[alias]) for alias in present) / len(present), 1)
    header_cells = ["Judge", *present, "Average"]
    value_cells = ["Score (%)", *(f"{scores[alias]:.1f}" for alias in present), f"{average:.1f}"]
    lines = [
        f"# Scores of judge models (prompt {template}, %)",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join([":---", *([":---:"] * (len(present) + 1))]) + " |",
        "| " + " | ".join(value_cells) + " |",
        "",
    ]
    path = scores_of_judges_md_path(output_root)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def default_gemma4_python_bin() -> Path:
    return _public_root() / "envs" / "gemma4" / ".venv" / "bin" / "python"


def resolve_gemma4_python_bin(args: argparse.Namespace) -> Path:
    python_bin = Path(args.gemma4_python_bin).expanduser().resolve() if args.gemma4_python_bin else default_gemma4_python_bin()
    if not python_bin.is_file():
        raise FileNotFoundError(
            f"Gemma 4 interpreter not found: {python_bin}. Set up the isolated "
            "environment first (see envs/gemma4/README.md), or pass "
            "--gemma4-python-bin to point at an existing interpreter."
        )
    return python_bin


def build_gemma4_command(args: argparse.Namespace, *, output_root: Path, python_bin: Path) -> list[str]:
    cmd = [
        str(python_bin),
        str(_public_root() / "scripts" / GEMMA4_SCRIPT_NAME),
        str(output_root),
        "--model-name",
        GEMMA4_JUDGE_ALIAS,
    ]
    if args.model_path:
        cmd.extend(["--model-path", args.model_path])
    if args.gpus_per_worker is not None:
        cmd.extend(["--gpus-per-worker", str(args.gpus_per_worker)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.dtype is not None:
        cmd.extend(["--dtype", args.dtype])
    if args.max_new_tokens is not None:
        cmd.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if args.do_sample:
        cmd.append("--do-sample")
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.eval_prompt is not None:
        cmd.extend(["--eval-prompt", args.eval_prompt])
    if args.ids:
        cmd.extend(["--ids", args.ids])
    if args.ids_file:
        cmd.extend(["--ids-file", args.ids_file])
    cmd.extend(["--dataset-dir", args.dataset_dir])
    return cmd


def run_gemma4_judge(args: argparse.Namespace, *, output_root: Path) -> int:
    python_bin = resolve_gemma4_python_bin(args)
    cmd = build_gemma4_command(args, output_root=output_root, python_bin=python_bin)
    print("=== Running local Gemma 4 judge evaluation ===")
    result = subprocess.run(cmd)
    return result.returncode


def run_api_judges(
    args: argparse.Namespace,
    *,
    output_root: Path,
    aliases: list[str],
    template_name: str,
    announce: bool,
) -> int:
    global MODEL
    env_config = load_env_config()

    eval_prompt_path = resolve_eval_prompt_path(args.eval_prompt)
    eval_prompt_config = load_eval_prompt_config(eval_prompt_path)
    _apply_eval_prompt_config(eval_prompt_config)

    sample_ids = resolve_sample_ids_from_args(args)
    dataset_root = price_dataset.resolve_dataset_root(args.dataset_dir)
    clients: dict[tuple[str, str], Any] = {}

    exit_code = 0
    for alias in aliases:
        route, provider = resolve_alias_route(alias, env_config, provider_mode=args.provider_mode)
        client_key = (route, provider)
        if client_key not in clients:
            clients[client_key] = build_client(route, provider, env_config)
        client = clients[client_key]

        MODEL = resolve_model_value(route, alias, provider, env_config)
        if announce:
            print(f"=== Evaluating with judge model: {alias} ({MODEL}, route={route}) ===")
        results, summary = run_evaluation(
            output_root,
            concurrency=args.concurrency,
            client=client,
            eval_prompt_path=eval_prompt_path,
            dataset_root=dataset_root,
            ids=sample_ids,
        )
        summary["route"] = route

        eval_results = {
            "evaluated_at": summary["evaluated_at"],
            "model": MODEL,
            "route": route,
            "output_root": str(output_root),
            "concurrency": args.concurrency,
            "samples": results,
        }

        output_paths = [
            eval_results_path(output_root, template=template_name, model=alias),
            eval_results_summary_path(output_root, template=template_name, model=alias),
        ]
        write_json(output_paths[0], eval_results)
        write_json(output_paths[1], summary)

        for path in output_paths:
            print(f"Wrote {path}")
        print(
            f"Done ({alias}, route={route}): {summary['successful_samples']}/{summary['total_samples']} "
            f"succeeded, average_score={summary['average_score']}"
        )
        if summary["failed_samples"] != 0:
            exit_code = 1

    return exit_code


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PRICE predictions with API VLM judges, optionally "
        "combined with the local Gemma 4 judge (see envs/gemma4/)."
    )
    parser.add_argument(
        "output_root",
        type=str,
        help="Output directory containing imgs/<sample_id>.png files.",
    )
    price_dataset.add_dataset_argument(parser)
    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        type=parse_judge_alias,
        metavar="{" + ",".join(ALL_JUDGE_ALIASES) + "}",
        help=f"Judge model alias (default: {DEFAULT_JUDGE_MODEL}). API aliases: "
        + ", ".join(
            f"{alias} ({API_JUDGE_MODEL_ALIASES[alias]})"
            for alias in ACTIVE_API_JUDGE_ALIASES
        )
        + f". Local alias: {GEMMA4_JUDGE_ALIAS} (requires envs/gemma4/, see its README).",
    )
    parser.add_argument(
        "--all-api-models",
        action="store_true",
        help="Evaluate with every API judge model in API_JUDGE_MODEL_ALIASES, writing one "
        "set of result files per model plus scores_of_judges.md. Overrides --model.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run the local Gemma 4 judge first, then every API judge, then write "
        "scores_of_judges.md. Overrides --model and --all-api-models.",
    )
    parser.add_argument(
        "--write-judge-scores-md",
        action="store_true",
        help="Write scores_of_judges.md from existing eval_results_summary files and exit.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests (default: 5).",
    )
    parser.add_argument(
        "--provider-mode",
        choices=PROVIDER_MODE_CHOICES,
        default="auto",
        help="How to route API judges (default: auto). auto: use a provider's official "
        "API (see env.toml [openai]/[gemini]/[doubao]) when a valid key is configured "
        "for that judge's provider, else fall back to [aihubmix]. official: require an "
        "official key for every judge run. aihubmix: always use the aihubmix aggregator, "
        "ignoring any configured official provider tables. MiniMax Flash requires "
        "[minimax] credentials and auto or official mode.",
    )
    parser.add_argument(
        "--eval-prompt",
        type=str,
        default=None,
        help="Path to eval prompt Python file. Defaults to eval_prompt/0601.py.",
    )
    add_ids_arguments(parser)

    local_group = parser.add_argument_group(
        "local Gemma 4 judge (used with --model gemma4, --all-models)"
    )
    local_group.add_argument(
        "--gemma4-python-bin",
        type=str,
        default=None,
        help="Interpreter used to run the local Gemma 4 judge (default: "
        "envs/gemma4/.venv/bin/python).",
    )
    local_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local Gemma 4 checkpoint path, forwarded as-is. If omitted, the judge "
        "script reads [gemma4].model_path from envs/gemma4/env.toml.",
    )
    local_group.add_argument("--gpus-per-worker", type=int, default=None)
    local_group.add_argument("--num-workers", type=int, default=None)
    local_group.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"], default=None
    )
    local_group.add_argument("--max-new-tokens", type=int, default=None)
    local_group.add_argument("--do-sample", action="store_true")
    local_group.add_argument("--temperature", type=float, default=None)

    return parser.parse_args(argv)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be >= 1")

    output_root = Path(args.output_root).expanduser().resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"output directory not found: {output_root}")

    eval_prompt_path = resolve_eval_prompt_path(args.eval_prompt)
    template_name = eval_prompt_template_name(eval_prompt_path)

    if args.write_judge_scores_md:
        path = write_scores_of_judges_md(output_root, template=template_name)
        print(f"Wrote {path}")
        return 0

    if args.all_models:
        run_gemma4 = True
        api_aliases = list(ACTIVE_API_JUDGE_ALIASES)
        write_md = True
    elif args.all_api_models:
        run_gemma4 = False
        api_aliases = list(ACTIVE_API_JUDGE_ALIASES)
        write_md = True
    elif args.model in LOCAL_JUDGE_ALIASES:
        run_gemma4 = True
        api_aliases = []
        write_md = False
    else:
        run_gemma4 = False
        api_aliases = [args.model]
        write_md = False

    exit_code = 0

    if run_gemma4:
        if run_gemma4_judge(args, output_root=output_root) != 0:
            exit_code = 1

    if api_aliases:
        announce = len(api_aliases) > 1
        if (
            run_api_judges(
                args,
                output_root=output_root,
                aliases=api_aliases,
                template_name=template_name,
                announce=announce,
            )
            != 0
        ):
            exit_code = 1

    if write_md:
        scores_path = write_scores_of_judges_md(output_root, template=template_name)
        print(f"Wrote {scores_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
