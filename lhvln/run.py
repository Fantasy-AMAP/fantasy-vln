import os
from utils.agent import HabitatAgent
from utils.metrics import NavigationMetrics
from utils.dataset import TaskDataset, EpisodeDataset, create_split_datasets
from utils.parser import read_args, random_seed
from torch.utils.data import DataLoader
from NavModel.RandomNav import RandomAgent
from NavModel.LLMModel.continuous_nav import ContinuousNav
from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch

import json
from NavModel.VLMModel import My_VLM_NavModel
from torch.utils.data import Subset


def split_indices(num_items: int, num_subsets: int):
    if num_items <= 0 or num_subsets <= 0 or num_subsets > num_items:
        raise ValueError("num_items and num_subsets must be positive, and num_subsets <= num_items")
    base_size, remainder = divmod(num_items, num_subsets)
    subsets = {}
    start = 0
    for i in range(num_subsets):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        subsets[f"subset_{i + 1}"] = [start, end]
        start = end
    return subsets


def validate_one_epoch(
        args, 
        epoch, 
        metrics, 
        dataloader, 
        nav_model,
        logger,
        subset_name
        ):
    
    nav_model.model.eval()

    num_batches_per_epoch = len(dataloader)
    total_training_steps = num_batches_per_epoch * args.num_epochs
    pbar = tqdm.tqdm(
        range(num_batches_per_epoch),
        disable=True,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch)
    )
    # test for LH-VLN task and step task 
    for step, (config, step_configs) in enumerate(dataloader):

        logger.info(config)
        # test for LH-task
        agent = HabitatAgent(args, config, nav_model, subset_name)
        result = agent.validate()
        metrics[str(len(result['successes']))].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
            )
        metrics[config["Robot"]].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
            )
        metrics['result'].add_sample(
            all(result['successes']),
            sum(result['gt_step']),
            sum(result['navigation_steps']),
            all(result['oracle_successes']),
            sum(result['navigation_errors'])/len(result['navigation_errors']) if len(result['navigation_errors']) > 0 else 0,
            result['successes'],
            result['gt_step'],
            result['gt_path'],
            result['navigation_errors']
                )

        verbose_dict = dict(
            step=step,
        )
        pbar.set_postfix(verbose_dict)
        pbar.update()
    

def main():
    args, global_cfg, logger, device_id = read_args()
    random_seed(args.seed)

    if args.tensorboard:
        writer_epoch = SummaryWriter(log_dir=args.tensorboard_path, filename_suffix='epoch')
    else:
        writer_epoch = None
    
    def custom_collate_fn(batch):
        if args.batch_size == 1:
            return batch[0]
        else:
            return batch
    
    if args.model_name == 'LLM Model':
        nav_model = ContinuousNav(args, global_cfg, logger, device_id)
    elif args.model_name == 'VLM Model':
        model_id = global_cfg.Model.pretrained_model_name_or_path if args.model_id is None else args.model_id
        
        if args.distributed:
            total_gpus = torch.cuda.device_count()
            device_id = args.rank % total_gpus
            nav_model = My_VLM_NavModel(model_id, device_id, use_ddp=True)
        else:
            nav_model = My_VLM_NavModel(model_id, device_id, use_ddp=False)
        
        if "worldvln" in model_id:
            nav_model.model_name = "WorldVLN"
    else:
        nav_model = RandomAgent()

    print(nav_model.model_name)

    if args.episode_data:
        train_dataset = EpisodeDataset(args, mode='train')
        val_dataset = EpisodeDataset(args, mode='valid')
        test_dataset = EpisodeDataset(args, mode='test') 
        # For unseen Test set
        if args.split_by_scene:
            train_dataset, val_dataset, test_dataset = create_split_datasets([train_dataset, val_dataset, test_dataset], args)    
    else:
        train_dataset = TaskDataset(args, mode='train')
        val_dataset = TaskDataset(args, mode='valid')
        test_dataset = TaskDataset(args, mode='test')
        # For unseen Test set
        if args.split_by_scene:
            train_dataset, val_dataset, test_dataset = create_split_datasets([train_dataset, val_dataset, test_dataset], args)
    
    subset_id = args.subset_id
    if subset_id == -1:
        subset_id = None
        subset_name = None
    if subset_id is not None:
        num_items = len(test_dataset)
        num_subset = args.num_gpus
        subset_name = f"subset_{subset_id}"
        [start_idx, end_idx] = split_indices(num_items, num_subset)[subset_name]
        indices = range(start_idx, end_idx)
        test_subset = Subset(test_dataset, indices)
    else:
        test_subset = test_dataset

    test_dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)

    test_metrics = {
        'result': NavigationMetrics(),
        '2': NavigationMetrics(),
        '3': NavigationMetrics(),
        '4': NavigationMetrics(),
        'spot': NavigationMetrics(),
        'stretch': NavigationMetrics(),
        'step': NavigationMetrics(),
    }

    validate_one_epoch(args, 0, test_metrics, test_dataloader, nav_model, logger, subset_name)
    logger.info("###### Test ######")
    for key, metrics in test_metrics.items():
        computed_metrics = metrics.compute()
        logger.info(f"Type: {key}")
        for metric_name, value in computed_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

    elsement_wise_results = {
        'result': {},
        '2': {},
        '3': {},
        '4':{},
        'spot': {},
        'stretch': {},
        'step': {},
    }

    for key in elsement_wise_results.keys():
        elsement_wise_results[key]['successes'] = test_metrics[key].successes
        elsement_wise_results[key]['gt_steps'] = test_metrics[key].gt_steps
        elsement_wise_results[key]['gt_length'] = test_metrics[key].gt_length
        elsement_wise_results[key]['error_length'] = test_metrics[key].error_length
        elsement_wise_results[key]['path_steps'] = test_metrics[key].path_steps
        elsement_wise_results[key]['oracle_successes'] = test_metrics[key].oracle_successes
        elsement_wise_results[key]['navigation_errors'] = test_metrics[key].navigation_errors
        elsement_wise_results[key]['subtask_successes'] = test_metrics[key].subtask_successes
        elsement_wise_results[key]['subtask_path_steps'] = test_metrics[key].subtask_path_steps
    
    if subset_name == None:
        save_path = "outputs/eval/results.txt" if args.save_path is None else args.save_path
    else:
        save_path = f"outputs/eval/{subset_name}_results.txt" if args.save_path is None else args.save_path
    
    if 'results' not in save_path:
        save_path = os.path.join(save_path, f'{subset_name}_results.txt')

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(elsement_wise_results, f, ensure_ascii=False, indent=4)

    if writer_epoch:
        writer_epoch.close()


if __name__ == '__main__':
    main()
