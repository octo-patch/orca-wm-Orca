from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import openai
from PIL import Image

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import eval_imgs  # noqa: E402


class ApiJudgesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.dataset = self.root / "dataset"
        split = self.dataset / "data" / "test"
        (split / "query").mkdir(parents=True)
        self.output = self.root / "predictions"
        (self.output / "imgs").mkdir(parents=True)
        Image.new("RGB", (2, 2), "red").save(split / "query" / "sample.jpg")
        Image.new("RGB", (2, 2), "blue").save(self.output / "imgs" / "sample.png")
        (split / "metadata.jsonl").write_text(json.dumps({
            "id": "sample", "query_file_name": "query/sample.jpg",
            "output_file_name": "query/sample.jpg",
            "lang": "move the object", "dataset": "example",
        }) + "\n", encoding="utf-8")
        self.requests: list[httpx.Request] = []
        self.sdk_class = openai.OpenAI
        for name in ("MODEL", "MIN_SCORE", "MAX_SCORE", "_eval_prompt_template"):
            self.addCleanup(setattr, eval_imgs, name, getattr(eval_imgs, name))

    def transport(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={
            "id": "offline-completion", "object": "chat.completion", "created": 0,
            "model": json.loads(request.content)["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": '{"score": 5, "reasoning": "ok"}',
            }}],
        })

    def make_client(self, **kwargs):
        client = self.sdk_class(
            **kwargs, max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.transport)),
        )
        self.addCleanup(client.close)
        return client

    def run_judge(self, config, *options):
        with (
            patch.object(eval_imgs, "load_env_config", return_value=config),
            patch.object(openai, "OpenAI", side_effect=self.make_client),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return eval_imgs.main([
                str(self.output), "--dataset-dir", str(self.dataset),
                "--concurrency", "1", *options,
            ])

    def assert_request(self, request, model, base_url):
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), base_url.rstrip("/") + "/chat/completions")
        body = json.loads(request.content)
        # Leave thinking/effort to the service defaults. In particular, never
        # send disabled thinking, reasoning_effort=none, or service_tier to Flash.
        self.assertEqual(set(body), {"model", "messages"})
        self.assertEqual(body["model"], model)
        self.assertEqual(body["messages"][0]["role"], "user")
        content = body["messages"][0]["content"]
        self.assertIn("move the object", content[0]["text"])
        images = [item["image_url"] for item in content if item["type"] == "image_url"]
        self.assertEqual(len(images), 2)
        for image in images:
            self.assertEqual(image["detail"], "high")
            prefix, encoded = image["url"].split(",", 1)
            self.assertEqual(prefix, "data:image/jpeg;base64")
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
                self.assertEqual(decoded.size, (2, 2))

    def test_direct_current_and_explicit_legacy_requests_preserve_endpoint_overrides(self):
        for alias, model in [
            ("minimax-m3.1-flash", "MiniMax-M3.1-Flash"),
            ("minimax-m3", "MiniMax-M3"),
            ("minimax-m2.7", "MiniMax-M2.7"),
        ]:
            for base_url in [None, "https://api.minimaxi.com/v1", "https://custom.invalid/api/v1"]:
                for mode in ("auto", "official"):
                    with self.subTest(alias=alias, base_url=base_url, mode=mode):
                        table = {"api_key": "offline-test-key"}
                        if base_url:
                            table["base_url"] = base_url
                        self.assertEqual(self.run_judge(
                            {"minimax": table}, "--model", alias, "--provider-mode", mode,
                        ), 0)
                        self.assert_request(self.requests[-1], model, base_url or "https://api.minimax.io/v1")
                        saved = json.loads((self.output / f"eval_results.0601.{alias}.json").read_text())
                        self.assertEqual(saved["model"], model)
                        self.assertEqual(saved["route"], "official")
                        self.assertEqual(saved["samples"][0]["status"], "success")

    def test_bulk_selection_prefers_flash_and_does_not_run_legacy(self):
        config = {
            "minimax": {"api_key": "offline-test-key"},
            "aihubmix": {"api_key": "offline-test-key"},
        }
        for option in ("--all-api-models", "--all-models"):
            with self.subTest(option=option), patch.object(eval_imgs, "run_gemma4_judge", return_value=0) as local:
                self.requests.clear()
                self.assertEqual(self.run_judge(config, option), 0)
                self.assertEqual([json.loads(r.content)["model"] for r in self.requests], [
                    "doubao-seed-2-0-pro", "gemini-3.1-pro-preview", "gpt-5.4-high",
                    "MiniMax-M3.1-Flash", "MiniMax-M3",
                ])
                self.assertEqual(local.call_count, int(option == "--all-models"))
                for request, model in zip(self.requests[-2:], ["MiniMax-M3.1-Flash", "MiniMax-M3"]):
                    self.assert_request(request, model, "https://api.minimax.io/v1")

    def test_flash_rejects_unverified_aggregator_before_any_request(self):
        for mode in ("auto", "official", "aihubmix"):
            for key in (None, "YOUR_MINIMAX_API_KEY", "offline-test-key"):
                if key == "offline-test-key" and mode != "aihubmix":
                    continue
                config = {"aihubmix": {"api_key": "offline-test-key"}}
                if key:
                    config["minimax"] = {"api_key": key}
                with self.subTest(mode=mode, key=key), self.assertRaisesRegex(ValueError, "requires a valid"):
                    self.run_judge(config, "--model", "minimax-m3.1-flash", "--provider-mode", mode)
        self.assertEqual(self.requests, [])

    def test_existing_aggregator_and_default_judge_requests_are_preserved(self):
        config = {"aihubmix": {"api_key": "offline-test-key"}}
        for alias, model in [("minimax-m3", "MiniMax-M3"), ("minimax-m2.7", "MiniMax-M2.7")]:
            for mode in ("auto", "aihubmix"):
                self.assertEqual(self.run_judge(config, "--model", alias, "--provider-mode", mode), 0)
                self.assert_request(self.requests[-1], model, eval_imgs.API_BASE_URL)
        self.assertEqual(self.run_judge(config), 0)
        self.assert_request(self.requests[-1], "gemini-3.1-pro-preview", eval_imgs.API_BASE_URL)

    def test_help_hides_legacy_but_explicit_alias_and_historical_scores_work(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as exit_info:
            eval_imgs.parse_args(["--help"])
        self.assertEqual(exit_info.exception.code, 0)
        self.assertNotIn("minimax-m2.7", output.getvalue())
        self.assertIn("minimax-m3.1-flash", output.getvalue())
        self.assertEqual(eval_imgs.parse_args(["out", "--model", "minimax-m2.7"]).model, "minimax-m2.7")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as invalid:
            eval_imgs.parse_args(["out", "--model", "misspelled-model"])
        self.assertEqual(invalid.exception.code, 2)
        history = self.output / "eval_results_summary.0601.minimax-m2.7.json"
        content = '{"model": "MiniMax-M2.7", "average_score_percentage": 80}'
        history.write_text(content, encoding="utf-8")
        self.assertEqual(self.run_judge({}, "--write-judge-scores-md"), 0)
        self.assertEqual(history.read_text(), content)
        self.assertIn("minimax-m2.7", (self.output / "scores_of_judges.md").read_text())
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
