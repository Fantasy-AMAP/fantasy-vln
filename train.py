import argparse
from typing import List, Optional, Union

from swift.ray import RayHelper
from swift.llm.train.sft import SwiftSft
from swift.llm.argument import TrainArguments
from swift.trainers import TrainerFactory
from swift.utils import get_logger, get_model_parameter_info
from swift.llm.infer import get_cached_dataset

from data.processor import load_dataset, UMMCoTDatasetLoader
from my_qwen_template import MyQwen2_5VLTemplate


logger = get_logger()


class MyTrainerFactory(TrainerFactory):
    TRAINER_MAPPING = {
        **TrainerFactory.TRAINER_MAPPING,
        'ummcot': 'trainer.MySeq2SeqTrainer',
    }

    TRAINING_ARGS_MAPPING = {
        **TrainerFactory.TRAINING_ARGS_MAPPING,
        'ummcot': 'swift.trainers.Seq2SeqTrainingArguments',
    }


class MySwiftSft(SwiftSft):
    def __init__(self, args: Optional[Union[List[str], TrainArguments]] = None) -> None:
        super().__init__(args)

    def _get_trainer_kwargs(self):
        kwargs = super()._get_trainer_kwargs()
        kwargs['use_ummcot'] = bool(getattr(self.args, 'use_ummcot', False))

        return kwargs
    
    def _get_dataset(self):
        # The random shuffling of the training set occurs in the dataloader of the trainer.
        args = self.args
        dataset_kwargs = args.get_dataset_kwargs()
        train_dataset, val_dataset = None, None
        if args.dataset:
            train_dataset, val_dataset = load_dataset(
                args.dataset,
                split_dataset_ratio=args.split_dataset_ratio,
                shuffle=args.dataset_shuffle,
                **dataset_kwargs)
        if len(args.val_dataset) > 0:
            # Loading val dataset
            _, val_dataset = load_dataset(
                args.val_dataset, split_dataset_ratio=1.0, shuffle=args.val_dataset_shuffle, **dataset_kwargs)
            assert args.split_dataset_ratio == 0.
        if args.truncation_strategy == 'split':
            logger.info(f'train_dataset: {train_dataset}')
            logger.info(f'val_dataset: {val_dataset}')
        return train_dataset, val_dataset
    
    @RayHelper.function(group='default')
    def _prepare_dataset(self):
        args = self.args
        # Defer encoding to the training phase
        pre_process = not (hasattr(args, 'rlhf_type') and args.rlhf_type in ['grpo', 'gkd'])
        if args.cached_dataset or args.cached_val_dataset:
            assert not args.streaming, 'Cached dataset does not support streaming.'
            train_datasets, val_datasets = get_cached_dataset(self.args)
        else:
            train_datasets, val_datasets = [], []
        if args.dataset or args.val_dataset:
            train_dataset, val_dataset = self._get_dataset()
            train_dataset, val_dataset = self._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)
            if train_dataset is not None:
                train_datasets.append(train_dataset)
            if val_dataset is not None:
                val_datasets.append(val_dataset)
        train_dataset = UMMCoTDatasetLoader._concat_datasets(train_datasets)
        val_dataset = UMMCoTDatasetLoader._concat_datasets(val_datasets)
        if args.truncation_strategy != 'split':
            logger.info(f'train_dataset: {train_dataset}')
            logger.info(f'val_dataset: {val_dataset}')
        datasets = [train_dataset, val_dataset]
        if not pre_process:
            return datasets
        datasets = self._post_process_datasets(datasets)
        self._show_dataset(*datasets)
        return datasets

    @RayHelper.function(group='default')
    def run(self):
        args = self.args
        train_dataset, val_dataset = self._prepare_dataset()
        
        if args.task_type == 'seq_cls':
            args.problem_type = args.problem_type or getattr(self.model.config, 'problem_type', None)
            logger.info(f'args.problem_type: {args.problem_type}')
        args.save_args()

        data_collator = self._get_data_collator()
        # Some tuners require train_dataset and data_collator for preparation: LoRA-GA
        self.model = self.prepare_model(self.args, self.model, template=self.template, train_dataset=train_dataset)
        logger.info(f'model: {self.model}')
        model_parameter_info = get_model_parameter_info(self.model)
        self.train_msg['model_parameter_info'] = model_parameter_info
        logger.info(f'model_parameter_info: {model_parameter_info}')
        
        if args.use_ummcot:
            args.task_type = 'ummcot'

        trainer_cls = MyTrainerFactory.get_trainer_cls(args)
        trainer = trainer_cls(
            model=self.model,
            args=self.args.training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=self.callbacks,
            template=self.template,
            **self._get_trainer_kwargs(),
        )
        return self.train(trainer)


def sft_main(args: Optional[Union[List[str], TrainArguments]] = None):
    return MySwiftSft(args).main()


def try_init_unsloth():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tuner_backend', type=str, default='peft')
    args, _ = parser.parse_known_args()

    if args.tuner_backend == 'unsloth':
        import unsloth  # noqa: F401


if __name__ == '__main__':
    from swift.cli.utils import try_use_single_device_mode
    from swift.ray import try_init_ray

    my_parser = argparse.ArgumentParser(add_help=False)
    my_parser.add_argument('--use_ummcot', action=argparse.BooleanOptionalAction, default=True, help='')
    my_args, remaining_argv = my_parser.parse_known_args()

    try_use_single_device_mode()
    try_init_unsloth()
    try_init_ray()

    from swift.llm import register_template
    from swift.llm.template.template.qwen import QwenTemplateMeta

    register_template(
        QwenTemplateMeta(
            MyQwen2_5VLTemplate.my_qwen2_5_vl,
            template_cls=MyQwen2_5VLTemplate
        )
    )

    sft = MySwiftSft(remaining_argv)
    sft.args.use_ummcot = my_args.use_ummcot
    sft.main()
