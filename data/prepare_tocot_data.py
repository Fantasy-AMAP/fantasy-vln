import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List, Optional

from data_utils import extract_substr, is_valid_think_block


def count_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def derive_output_path(input_jsonl: str, output_jsonl: Optional[str]) -> str:
    if output_jsonl is not None and output_jsonl.strip():
        return output_jsonl
    # keep your original convention: swift_ -> tcot_swift_
    return input_jsonl.replace("swift_", "tcot_swift_")


def prepare_tcot_dataset(
    excel_path: str,
    input_jsonl: str,
    output_jsonl: Optional[str] = None,
    success_col: str = "是否成功",
    success_value: str = "成功",
    answer_col: str = "回答",
    replace_from: str = "no_textual_think",
    replace_to: str = "textual_think",
    strict_row_alignment: bool = True,
) -> List[int]:
    assert os.path.isfile(excel_path), f"Excel not found: {excel_path}"
    assert os.path.isfile(input_jsonl), f"JSONL not found: {input_jsonl}"

    out_path = derive_output_path(input_jsonl, output_jsonl)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    anno_res = pd.read_excel(excel_path)

    for col in (success_col, answer_col):
        if col not in anno_res.columns:
            raise KeyError(f"Column '{col}' not found in excel. Available: {list(anno_res.columns)}")

    total = count_lines(input_jsonl)
    if strict_row_alignment and len(anno_res) < total:
        raise ValueError(
            f"Excel rows ({len(anno_res)}) < JSONL lines ({total}). "
            "Your original logic assumes 1-to-1 row alignment by index."
        )

    invalid_lines: List[int] = []
    kept = 0
    skipped_not_success = 0

    with open(input_jsonl, "r", encoding="utf-8") as infile, open(out_path, "w", encoding="utf-8") as outfile:
        for i, line in enumerate(tqdm(infile, total=total, desc="Processing")):
            line = line.strip()
            if not line:
                invalid_lines.append(i)
                continue

            # alignment guard (optional)
            if strict_row_alignment and i >= len(anno_res):
                raise ValueError(
                    f"Line index {i} out of excel range (rows={len(anno_res)}). "
                    "Disable --no_strict_row_alignment if you really want to proceed."
                )

            data = json.loads(line)

            # filter by excel success flag
            if str(anno_res[success_col].iloc[i]) != success_value:
                skipped_not_success += 1
                continue

            # replace tag in user message
            try:
                data["messages"][0]["content"] = data["messages"][0]["content"].replace(replace_from, replace_to)
            except Exception:
                invalid_lines.append(i)
                continue

            # original gt
            try:
                gt = data["messages"][1]["content"]
            except Exception:
                invalid_lines.append(i)
                continue

            tcot = anno_res[answer_col].iloc[i]

            if not is_valid_think_block(tcot):
                invalid_lines.append(i)
                continue

            new_gt = f"{tcot}<var></var><answer>{extract_substr(gt)}</answer>"
            data["messages"][1]["content"] = new_gt

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[Done] output: {out_path}")
    print(f"[Stats] total_lines={total}, kept={kept}, skipped_not_success={skipped_not_success}, illegal_num={len(invalid_lines)}")
    if invalid_lines:
        print("[Illegal line indices]")
        print(invalid_lines)

    return invalid_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare textual-CoT (tCoT) jsonl from excel annotations.")
    parser.add_argument("--excel_path", type=str, required=False, help="Path to excel annotation file.", default="data/lhvln_tcot_anno.xlsx")
    parser.add_argument("--input_jsonl", type=str, required=False, help="Path to input swift jsonl file.", default="data/json_files/swift_his_20_val.jsonl")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Output path. Default: replace swift_ -> tcot_swift_.")
    parser.add_argument("--success_col", type=str, default="是否成功", help="Excel column name for success flag.")
    parser.add_argument("--success_value", type=str, default="成功", help="Excel cell value indicating success.")
    parser.add_argument("--answer_col", type=str, default="回答", help="Excel column name for annotated tCoT answer.")
    parser.add_argument("--replace_from", type=str, default="no_textual_think", help="String to replace in user prompt.")
    parser.add_argument("--replace_to", type=str, default="textual_think", help="Replacement string in user prompt.")
    parser.add_argument(
        "--no_strict_row_alignment",
        action="store_true",
        help="Disable strict check that excel rows align with jsonl line indices.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    prepare_tcot_dataset(
        excel_path=args.excel_path,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        success_col=args.success_col,
        success_value=args.success_value,
        answer_col=args.answer_col,
        replace_from=args.replace_from,
        replace_to=args.replace_to,
        strict_row_alignment=(not args.no_strict_row_alignment),
    )
