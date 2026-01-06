import ast
import csv
from pathlib import Path


MODELS = {
    "gpt-5.2": "gpt52",
    "gpt-5-mini": "gpt5mini",
    "gpt-5-nano": "gpt5nano",
    "deepseek-chat": "deepseekchat",
}

METHODS = {
    "ReAct": "artifacts/output/travel/baseline/{model}_validation/results.txt",
    "MemPlan": "artifacts/output/travel/{model}_validation/results.txt",
    "MemPlan-NoRepair": "artifacts/output/travel/{model}_validation/raw_results.txt",
}

LEVELS = ["easy", "medium", "hard"]


def extract_summary_dict(text: str) -> dict:
    start = text.rfind("{'Commonsense Constraint'")
    if start == -1:
        raise ValueError("Summary dict not found.")
    brace_count = 0
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                end = idx + 1
                break
    if end is None:
        raise ValueError("Unbalanced braces in summary dict.")
    return ast.literal_eval(text[start:end])


def budget_counts(file_path: Path) -> dict:
    text = file_path.read_text(encoding="utf-8")
    summary = extract_summary_dict(text)
    hard = summary["Hard Constraint"]
    counts = {level: {"true": 0, "total": 0} for level in LEVELS}
    for level in LEVELS:
        for day_data in hard[level].values():
            budget = day_data.get("Budget")
            if not budget:
                continue
            counts[level]["true"] += budget.get("true", 0)
            counts[level]["total"] += budget.get("total", 0)
    counts["all"] = {
        "true": sum(counts[level]["true"] for level in LEVELS),
        "total": sum(counts[level]["total"] for level in LEVELS),
    }
    return counts


def safe_rate(true_count: int, total_count: int) -> float:
    return true_count / total_count if total_count else 0.0


def main() -> None:
    output_path = Path("artifacts/output/travel/budget_satisfaction.csv")
    rows = []
    for model_name, model_key in MODELS.items():
        method_rates = {}
        for method, template in METHODS.items():
            file_path = Path(template.format(model=model_key))
            counts = budget_counts(file_path)
            method_rates[method] = {
                level: safe_rate(counts[level]["true"], counts[level]["total"])
                for level in ["easy", "medium", "hard", "all"]
            }
        for level in ["easy", "medium", "hard", "all"]:
            row_id = f"{level}_{model_name}"
            rows.append(
                {
                    "set_model": row_id,
                    "ReAct": method_rates["ReAct"][level],
                    "MemPlan": method_rates["MemPlan"][level],
                    "MemPlan-NoRepair": method_rates["MemPlan-NoRepair"][level],
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["set_model", "ReAct", "MemPlan", "MemPlan-NoRepair"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
