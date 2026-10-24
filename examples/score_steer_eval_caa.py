#!/usr/bin/env python3
"""给已经生成的 CAA 文本打分，并汇总各领域、各强度的平均分。

在 examples/ 下运行：

    export API_KEY=你的密钥
    export BASE_URL=https://api.deepseek.com   # 不设则默认这个地址
    python score_steer_eval_caa.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent

DEFAULT_DATASETS = [
    "language_features",
    "personality",
    "reasoning_patterns",
    "sentiment",
]
DEFAULT_MULTIPLIERS = [1, 2, 3, 4, 5, 6, 7, 8]
DEFAULT_BASE_URL = "https://api.deepseek.com"
SCORE_KEYS = (
    "all_average_concept_scores",
    "all_average_instruction_scores",
    "all_average_fluency_scores",
    "all_average_scores",
)
SCORE_LABELS = {
    "all_average_concept_scores": "concept",
    "all_average_instruction_scores": "instruction",
    "all_average_fluency_scores": "fluency",
    "all_average_scores": "harmonic",
}


def parse_args():
    parser = argparse.ArgumentParser(description="用 DeepSeek 给已生成的 CAA 文本打分")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--multipliers", nargs="+", type=int, default=DEFAULT_MULTIPLIERS)
    parser.add_argument("--judge", default="deepseek-flash")
    parser.add_argument("--model", default="gemma-2-2b-it")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--gen-out-path", default="f_steer_eval")
    return parser.parse_args()


def generation_dir(model, dataset, gen_out, layer, multiplier):
    return (
        EXAMPLES_DIR
        / "generation"
        / model
        / "SteerEval"
        / dataset
        / gen_out
        / "caa"
        / f"layer_{layer}_multip_{multiplier}"
    )


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pick_scores(data):
    return {key: data[key] for key in SCORE_KEYS}


def ensure_results_link(directory):
    link = directory / "all_generation_results.json"
    if link.exists() or link.is_symlink():
        return
    link.symlink_to("all_generation_results_valid.json")


def get_evaluator():
    if not os.environ.get("API_KEY"):
        sys.exit("请先设置环境变量 API_KEY，再运行评分。")
    os.environ.setdefault("BASE_URL", DEFAULT_BASE_URL)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from steer.utils.steer_eval_utils import evaluate_steer_eval
    return evaluate_steer_eval


def classify(directory, judge):
    valid_path = directory / "all_generation_results_valid.json"
    scored_path = directory / f"all_evaluation_results_{judge}.json"
    row = {"dir": str(directory.relative_to(EXAMPLES_DIR))}
    if not valid_path.is_file():
        row["status"] = "missing"
        return row, None
    if scored_path.is_file():
        row["status"] = "skipped"
        row.update(pick_scores(load_json(scored_path)))
        return row, scored_path
    try:
        payload = load_json(valid_path)
    except json.JSONDecodeError as exc:
        row["status"] = "error"
        row["error"] = f"无法读取生成结果: {exc}"
        return row, None
    if not isinstance(payload, list) or len(payload) == 0:
        row["status"] = "empty"
        return row, None
    row["status"] = "pending"
    return row, scored_path


def print_table(rows):
    print()
    current = None
    for row in rows:
        if row["dataset"] != current:
            current = row["dataset"]
            print(current)
        scores = "  ".join(
            f"{label}={row[key]:.4f}" if key in row else f"{label}=-    "
            for key, label in SCORE_LABELS.items()
        )
        print(f"  multip {row['multiplier']:>2}  {row['status']:<8}  {scores}")


def main():
    args = parse_args()
    os.environ.setdefault("BASE_URL", DEFAULT_BASE_URL)

    pending = []
    rows = []
    for dataset in args.datasets:
        for multiplier in args.multipliers:
            directory = generation_dir(
                args.model, dataset, args.gen_out_path, args.layer, multiplier
            )
            row, scored_path = classify(directory, args.judge)
            row["dataset"] = dataset
            row["multiplier"] = multiplier
            rows.append(row)
            if row["status"] == "pending":
                pending.append((row, directory, scored_path))
            elif row["status"] == "missing":
                print(f"跳过 {dataset} multiplier {multiplier}：没有 all_generation_results_valid.json")
            elif row["status"] == "empty":
                print(f"跳过 {dataset} multiplier {multiplier}：生成结果为空")
            elif row["status"] == "skipped":
                print(f"已有评分，跳过 {dataset} multiplier {multiplier}")
            elif row["status"] == "error":
                print(f"跳过 {dataset} multiplier {multiplier}：{row['error']}")

    evaluate = get_evaluator() if pending else None
    for row, directory, scored_path in pending:
        dataset = row["dataset"]
        multiplier = row["multiplier"]
        try:
            ensure_results_link(directory)
            print(f"评分 {dataset} multiplier {multiplier} ...")
            evaluate(args.judge, str(directory))
            row["status"] = "scored"
            row.update(pick_scores(load_json(scored_path)))
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
            print(f"评分失败 {dataset} multiplier {multiplier}: {exc}")

    summary_path = (
        EXAMPLES_DIR
        / "generation"
        / args.model
        / "SteerEval"
        / f"caa_scores_{args.judge}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "judge": args.judge,
        "model": args.model,
        "layer": args.layer,
        "base_url": os.environ.get("BASE_URL"),
        "rows": rows,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print_table(rows)
    print(f"\n汇总已写入 {summary_path.relative_to(EXAMPLES_DIR)}")


if __name__ == "__main__":
    main()
