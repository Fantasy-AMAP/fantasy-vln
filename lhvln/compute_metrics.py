import os
import json
import argparse

from utils.metrics import NavigationMetrics


def merge_json_list(json_list):
    if not all(isinstance(d, dict) for d in json_list):
        raise ValueError("所有元素必须是字典")

    keys = list(json_list[0].keys())
    for d in json_list[1:]:
        if list(d.keys()) != keys:
            raise ValueError("JSON 结构不一致")

    merged = {}
    for k in keys:
        values = [d[k] for d in json_list]
        if all(isinstance(v, dict) for v in values):
            merged[k] = merge_json_list(values)
        elif all(isinstance(v, list) for v in values):
            merged[k] = sum(values, [])
        else:
            merged[k] = values
    return merged


def main(args):
    res_dir = args.res_dir
    files = os.listdir(res_dir)
    res_path = []
    for file in files:
        if 'results' in file:
            res_path.append(os.path.join(res_dir, file))


    json_list = []
    for path in res_path:
        with open(path, "r", encoding="utf-8") as f:
            json_list.append(json.load(f))
    res = merge_json_list(json_list)

    print(len(res['result']['successes']))
    print(len(res['2']['successes']) * 2 + len(res['3']['successes']) * 3 + len(res['4']['successes']) * 4)

    test_metrics = {
        'result': NavigationMetrics(),
        '2': NavigationMetrics(),
        '3': NavigationMetrics(),
        '4': NavigationMetrics(),
        'spot': NavigationMetrics(),
        'stretch': NavigationMetrics(),
        'step': NavigationMetrics(),
    }

    for key in res.keys():
        test_metrics[key].successes = res[key]['successes']
        test_metrics[key].gt_steps = res[key]['gt_steps']
        test_metrics[key].gt_length = res[key]['gt_length']
        test_metrics[key].error_length = res[key]['error_length']
        test_metrics[key].path_steps = res[key]['path_steps']
        test_metrics[key].oracle_successes = res[key]['oracle_successes']
        test_metrics[key].navigation_errors = res[key]['navigation_errors']
        test_metrics[key].subtask_successes = res[key]['subtask_successes']
        test_metrics[key].subtask_path_steps = res[key]['subtask_path_steps']

    for key, metrics in test_metrics.items():
        computed_metrics = metrics.compute()
        print(f"Type: {key}")
        for metric_name, value in computed_metrics.items():
            print(f"  {metric_name}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
