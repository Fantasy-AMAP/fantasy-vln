import json
import copy
import argparse
from pathlib import Path

from data_utils import extract_substr


def load_lines(str_path):
    p = Path(str_path)
    with p.open("r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def check_data_alignment(data1, data2):
    keys = ['images', 'future_images', 'batch_id', 'sub_task_num', 'task_name', 'end_act_id']
    return all(data1.get(k) == data2.get(k) for k in keys)


def tvcot_to_mmcot_data(v_data, t_data):
    t_cot = extract_substr(
        t_data['messages'][1]['content'],
        "<think>",
        "</think>"
    )

    mm_data = copy.deepcopy(v_data)
    mm_data['messages'][1]['content'] = mm_data['messages'][1]['content'].replace(
        "<think></think>",
        f"<think>{t_cot}</think>"
    )
    mm_data['messages'][0]['content'] = mm_data['messages'][0]['content'].replace(
        "<no_textual_think>",
        "<textual_think>"
    )
    return mm_data

def tcot_to_noncot_data(t_data):
    answer = extract_substr(
        t_data['messages'][1]['content'],
        "<answer>",
        "</answer>"
    )
    non_cot_data = copy.deepcopy(t_data)
    non_cot_data['messages'][1]['content'] = f"<think></think><var></var><answer>{answer}</answer>"
    non_cot_data['messages'][0]['content'] = non_cot_data['messages'][0]['content'].replace(
        "<textual_think>",
        "<no_textual_think>"
    )
    return non_cot_data

def main(vcot_json_path: str, tcot_json_path: str, mmcot_json_path: str, save_as_ummcot_format: bool):
    vcot_lines = load_lines(vcot_json_path)
    tcot_lines = load_lines(tcot_json_path)

    vi, ti = 0, 0

    with open(mmcot_json_path, "w", encoding="utf-8") as mmcot_file:
        while ti < len(tcot_lines) and vi < len(vcot_lines):
            v_data = json.loads(vcot_lines[vi])
            t_data = json.loads(tcot_lines[ti])

            if check_data_alignment(v_data, t_data):
                mm_data = tvcot_to_mmcot_data(v_data, t_data)
                if not save_as_ummcot_format:
                    mmcot_file.write(json.dumps(mm_data, ensure_ascii=False) + "\n")
                else:
                    non_cot_data = tcot_to_noncot_data(t_data)
                    ummcot_data = {
                        "Non_CoT": non_cot_data,
                        "T_CoT": t_data,
                        "V_CoT": v_data,
                        "MM_CoT": mm_data
                    }
                    mmcot_file.write(json.dumps(ummcot_data, ensure_ascii=False) + "\n")
                vi += 1
                ti += 1
            else:
                vi += 1
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge vcot and tcot jsonl into mmcot jsonl (strictly equivalent logic)."
    )
    parser.add_argument(
        "--vcot_json_path",
        type=str,
        required=False,
        help="Path to vcot jsonl file.",
        default="data/json_files/vcot_swift_his_20_val.jsonl"
    )
    parser.add_argument(
        "--tcot_json_path",
        type=str,
        required=False,
        help="Path to tcot jsonl file.",
        default="data/json_files/tcot_swift_his_20_val.jsonl"
    )
    parser.add_argument(
        "--mmcot_json_path",
        type=str,
        default=None,
        help="Output mmcot jsonl path. Default: replace 'vcot' with 'mmcot' in vcot path."
    )
    parser.add_argument(
        "--save_as_ummcot_format",
        type=bool,
        default=True,
        help=""
    )

    args = parser.parse_args()

    if args.mmcot_json_path is None:
        mmcot_json_path = args.vcot_json_path.replace("vcot", "mmcot")
        if args.save_as_ummcot_format:
            mmcot_json_path = mmcot_json_path.replace("mmcot", "ummcot")
    else:
        mmcot_json_path = args.mmcot_json_path

    main(
        vcot_json_path=args.vcot_json_path,
        tcot_json_path=args.tcot_json_path,
        mmcot_json_path=mmcot_json_path,
        save_as_ummcot_format=args.save_as_ummcot_format
    )
