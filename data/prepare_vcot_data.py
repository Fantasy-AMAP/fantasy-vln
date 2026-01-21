import os
import json
import torch
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from data_utils import extract_substr

MAX_WORKERS = min(32, (os.cpu_count() or 8))
SUBSET_NAME = True

def ipath_to_varpath(img_path: str) -> str:
    return img_path.replace("/task/", "/var_tokens/").replace(".png", ".pt")


def process_one_line(line: str, scale_schedule: list) -> str:
    data = json.loads(line.strip())

    future_imgs = data['future_images']
    gt = extract_substr(data['messages'][1]['content'], "<answer>", "</answer>")

    future_var_tokens = []

    for img_path in future_imgs:
        var_path = ipath_to_varpath(img_path)
        scale = scale_schedule[0]
        arr = torch.load(var_path, map_location='cpu')[scale].tolist()
        var_token = [f"<|{token_id}|>" for token_id in arr]
        var_token = "".join(var_token)
        future_var_tokens.append(var_token)

    new_gt = "<think></think>"
    for var_token in future_var_tokens:
        new_gt += f"<var>{var_token}</var>"
    new_gt += f"<answer>{gt}</answer>"

    data['messages'][1]['content'] = new_gt
    data['messages'][0]['content'] = data['messages'][0]['content'].replace("no_visual_think", "visual_think")

    return json.dumps(data, ensure_ascii=False)

def count_lines(path: str) -> int:
    with open(path, "r") as f:
        return sum(1 for _ in f)

def main(scale_schedule, input_file_path, output_file_path):
    total_line_num = count_lines(input_file_path)

    with open(input_file_path, "r") as infile, open(output_file_path, "w") as outfile:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for out_line in tqdm(
                executor.map(process_one_line, infile, [scale_schedule] * total_line_num),
                total=total_line_num,
                desc="Processing"
            ):
                outfile.write(out_line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_schedule", type=int, nargs="+", default=[3])
    parser.add_argument(
        "--input_file_path",
        type=str,
        default="data/json_files/swift_his_20_val.jsonl",
    )
    parser.add_argument(
        "--subset_name",
        action="store_true",
        default=False,
        help="Keep original behavior (default: False). If set, output filename contains var scale id.",
    )
    parser.add_argument(
        "--no_subset_name",
        action="store_true",
        help="Disable subset_name behavior (equivalent to SUBSET_NAME=False).",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        default=None,
        help="If not set, derived from input_file_path using the original replace logic.",
    )

    args = parser.parse_args()

    assert isinstance(args.scale_schedule, list)
    scale_schedule = args.scale_schedule

    subset_name = args.subset_name and (not args.no_subset_name)

    input_file_path = args.input_file_path

    if args.output_file_path is not None:
        output_file_path = args.output_file_path
    else:
        if subset_name:
            output_file_path = input_file_path.replace("swift_", f"vcot_swift_var_{scale_schedule[0]}_")
        else:
            output_file_path = input_file_path.replace("swift_", f"vcot_swift_")

    main(scale_schedule=scale_schedule, input_file_path=input_file_path, output_file_path=output_file_path)
