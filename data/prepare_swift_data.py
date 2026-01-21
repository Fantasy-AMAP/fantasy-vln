import re
import os
import json
import math
import random
import argparse
from tqdm import tqdm
from math import log2
from typing import List, Optional
from collections import defaultdict


def sample_subset(strings: List[str], ratio: float, *, seed: Optional[int] = None) -> List[str]:
    assert isinstance(strings, list)
    assert all(isinstance(s, str) for s in strings)
    n = len(strings)
    assert n > 0, "The input strings can not be empty!"
    assert isinstance(ratio, (int, float))
    assert 0.0 <= ratio <= 1.0

    k = max(1, math.ceil(n * ratio))
    if seed is not None:
        rnd = random.Random(seed)
        chosen_indices = rnd.sample(range(n), k)
    else:
        chosen_indices = random.sample(range(n), k)

    chosen_indices.sort()
    return [strings[i] for i in chosen_indices]

def vln_action_complexity(seq_str: str, alpha: float = 0.5, window_size=5) -> float:
    actions=("forward","left","right","stop")
    toks=[m.group(1).lower() for m in re.finditer(r"<\|([a-z]+)\|?>",seq_str)]
    if any(t not in actions for t in toks): raise ValueError
    if len(toks)>window_size: raise ValueError
    if "stop" in toks:
        if toks[-1]!="stop" or toks.count("stop")!=1: raise ValueError
    else:
        if len(toks)!=window_size: raise ValueError
    n,K=len(toks),4
    counts=defaultdict(lambda:{a:0 for a in actions})
    L,ctx=0.0,"<BOS>"
    for a in toks:
        table=counts[ctx]; total=sum(table.values())
        p=(table[a]+alpha)/(total+alpha*K)
        L-=log2(p)
        counts[ctx][a]+=1
        ctx=a
    s=max(0.0,min(1.0,L/(n*log2(K))))
    s*=n/(n+2)
    idx=len(toks)-1 if toks[-1]=="stop" else len(toks)
    trans=max(0,idx-1)
    changes=sum(1 for i in range(1,idx) if toks[i]!=toks[i-1])
    r=changes/trans if trans>0 else 0.0
    return max(0.0,min(1.0,0.6*s+0.4*r))


def build_user_prompt(user_instruction: str, history_count: int, stop_count: int, window_size: int) -> str:
    lines = []
    lines.append(
        "<no_textual_think><no_visual_think>You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.\n"
        f"Based on this information, you need to decide your next {window_size} actions, which could involve <|left|>,<|right|>,<|forward|>. "
        "If you finish your mission, output <|stop|>. Here are some examples: <|left|><|forward|><|forward|><|stop|>, "
        "<|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>"
    )
    if history_count > 0:
        lines.append(f"# Your historical pictures are: {'<image>' * history_count}")
    lines.append("# Your current observations are left side: <image>, front side: <image>, right side: <image>")
    lines.append(f"# Your mission is: {user_instruction}")
    lines.append(
        "PS: The mission is complex. You may infer several sub-tasks within the mission, and output <|stop|> when a sub-task is achieved. "
        f"So far, you have output <|stop|> {stop_count} times. Historical information reflects progress up to the current subgoal. <|NAV|>"
    )
    return "\n".join(lines)


def str_to_action(input_str: str) -> str:
    s = input_str.lower()
    if "forward" in s:
        return "<|forward|>"
    elif "left" in s:
        return "<|left|>"
    elif "right" in s:
        return "<|right|>"
    elif "stop" in s:
        return "<|stop|>"
    else:
        raise ValueError(f"Illegal action: {input_str}")


def generate_swift_dataset(args):
    # Get parameters
    set_name = args.set_name
    base_dir = args.base_dir
    max_his_image_num = args.max_his_image_num
    window_size = args.window_size
    data_augmentation = args.data_augmentation
    sample_ratio = args.sample_ratio
    output_dir = args.output_dir


    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    if not data_augmentation:
        output_file = os.path.join(output_dir, f"swift_his_20_{set_name}.jsonl")
    else:
        output_file = os.path.join(output_dir, f"swift_his_20_{set_name}_aug.jsonl")
    
    assert 0. < sample_ratio <= 1.
    if sample_ratio < 1.:
        output_file = output_file.replace("swift_his_20", "sub_swift_his_20")

    # Get batch ids
    if set_name == "train":
        batch_ids = [1, 2, 3, 4, 5]
    elif set_name == "test":
        batch_ids = [7, 8]
    elif set_name == "val":
        batch_ids = [6]
    else:
        raise ValueError(f"set_name must be one of: 'train', 'test', 'val', but got {set_name}.")

    # loopping for data generation
    with open(output_file, "w", encoding="utf-8") as outfile:
        for i in batch_ids:
            batch_path = os.path.join(base_dir, f"batch_{i}")

            for j in ["2", "3", "4"]:
                task_dir = os.path.join(batch_path, j)
                assert os.path.isdir(task_dir)

                task_instructions = os.listdir(task_dir)

                if sample_ratio < 1.:
                    task_instructions = sample_subset(task_instructions, sample_ratio, seed=42)

                for ins in tqdm(task_instructions, desc=f"[{i}/8][{int(j) - 1}/3]"):
                    trial_path = os.path.join(batch_path, j, ins, "success", "trial_1")
                    assert os.path.isdir(trial_path)

                    entries = os.listdir(trial_path)
                    action_dirs = []
                    for name in entries:
                        full = os.path.join(trial_path, name)
                        if not os.path.isdir(full):
                            continue
                        m = re.match(r"^(-?\d+)_", name)
                        if m:
                            action_dirs.append((int(m.group(1)), name))

                    assert len(action_dirs) > 0

                    action_dirs.sort(key=lambda x: x[0])

                    if data_augmentation:
                        round = window_size
                    else:
                        round = 1
                    
                    for k in range(round):
                        images = []
                        action_list = []
                        future_images = []
                        act_num = 0
                        stop_count = 0

                        for act_idx, act_str in action_dirs:
                            act_path = os.path.join(trial_path, act_str)

                            # import pdb; pdb.set_trace()
                            if act_idx <= k - 1:
                                if act_idx == k - 1:
                                    images.append(os.path.join(act_path, "left.png"))
                                    images.append(os.path.join(act_path, "front.png"))
                                    images.append(os.path.join(act_path, "right.png"))
                                continue

                            act = str_to_action(act_str)
                            act_num += 1
                            action_list.append(act)
                            future_images.append(os.path.join(act_path, "front.png"))

                            if act_num == window_size or act == "<|stop|>":
                                assert stop_count <= 3
                                assert act_num == len(future_images)

                                if len(images) > max_his_image_num + 3:
                                    images = images[-(max_his_image_num + 3):]

                                history_count = max(0, len(images) - 3)
                                usr_prompt = build_user_prompt(
                                    user_instruction=ins, 
                                    history_count=history_count, 
                                    stop_count=stop_count, 
                                    window_size=window_size
                                )
                                gt_acts = ''.join(action_list)
                                vln_answer = f"<think></think><var></var><answer>{gt_acts}</answer>"

                                data = {
                                    "messages": [
                                        {
                                            "content": usr_prompt,
                                            "role": "user",
                                        },
                                        { 
                                            "content": vln_answer,
                                            "role": "assistant",
                                        },
                                    ],
                                    "images": images,
                                    # additional items
                                    "future_images": future_images,
                                    "batch_id": i,
                                    "sub_task_num": int(j),
                                    "task_name": ins,
                                    "end_act_id": act_idx,
                                }

                                if k == 0:
                                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                                else:
                                    if vln_action_complexity(gt_acts, window_size=window_size) > 0.7 or '<|stop|>' in gt_acts:
                                        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

                                front_image = images[-2]
                                images = images[:-3]
                                images.append(front_image)

                                images.extend(future_images[:-1])

                                if act == "<|stop|>":
                                    stop_count += 1
                                    images = []

                                images.append(os.path.join(act_path, "left.png"))
                                images.append(os.path.join(act_path, "front.png"))
                                images.append(os.path.join(act_path, "right.png"))

                                act_num = 0
                                action_list = []
                                future_images = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_name", type=str, default="val", choices=["train", "test", "val"], help="Dataset split to use.")
    # TODO: change default value
    parser.add_argument("--base_dir", type=str, default="data/images", help="Base directory for LH-VLN data.")
    parser.add_argument("--max_his_image_num", type=int, default=20, help="Maximum number of historical images to include.")
    parser.add_argument("--window_size", type=int, default=5, help="Number of future actions the model predicts in each sliding window step.")
    parser.add_argument("--data_augmentation", action="store_true", help="Enable data augmentation during training (default: disabled).")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Proportion of the original dataset to sample. 1.0 means using the full dataset.")
    parser.add_argument("--output_dir", type=str, default="./data/json_files", help="Output directory for generated qwen format json files.")

    args = parser.parse_args()

    generate_swift_dataset(args)
