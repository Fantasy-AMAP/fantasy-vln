import os
import numpy as np
from typing import Dict, List, Literal, Optional, Tuple, Union, Any

from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset

from modelscope.hub.utils.utils import get_cache_dir

from swift.utils import get_logger, get_seed, safe_ddp_context, use_hf_hub
from swift.llm.dataset.register import DATASET_MAPPING, DATASET_TYPE, DatasetMeta
from swift.llm.dataset.loader import DatasetLoader, init_self_cognition_preprocessor, DatasetSyntax, SubsetDataset
from swift.llm import MessagesPreprocessor, DatasetMeta, register_dataset


DATASET_TYPE = Union[HfDataset, HfIterableDataset]


logger = get_logger()


class UMMCoTDatasetLoader(DatasetLoader):
    @staticmethod
    def _load_dataset_path(
        dataset_path: str,
        dataset_meta: DatasetMeta,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        strict: bool = False,
        streaming: bool = False,
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ) -> HfDataset:
        ext = os.path.splitext(dataset_path)[1].lstrip('.')
        file_type = {'jsonl': 'json', 'txt': 'text'}.get(ext) or ext
        kwargs = {'split': 'train', 'streaming': streaming, 'num_proc': num_proc}
        if file_type == 'csv':
            kwargs['na_filter'] = False
        with safe_ddp_context(None, True):
            kwargs['cache_dir'] = os.path.join(get_cache_dir(), 'datasets')
            dataset = hf_load_dataset(file_type, data_files=dataset_path, **kwargs)
        if columns:
            dataset = UMMCOTPreprocessor.safe_rename_columns(dataset, columns)
        dataset = dataset_meta.preprocess_func(
            dataset, num_proc=num_proc, load_from_cache_file=load_from_cache_file, strict=strict)
        if remove_unused_columns:
            dataset = UMMCOTPreprocessor.remove_useless_columns(dataset)
        return dataset
    
    @staticmethod
    def load(
        dataset_syntax: Optional[DatasetSyntax] = None,
        dataset_meta: Optional[DatasetMeta] = None,
        *,
        num_proc: int = 1,
        load_from_cache_file: bool = True,
        streaming: bool = False,
        use_hf: Optional[bool] = None,
        hub_token: Optional[str] = None,
        strict: bool = False,
        download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
        columns: Optional[Dict[str, str]] = None,
        remove_unused_columns: bool = True,
    ) -> HfDataset:
        if dataset_syntax.dataset_type == 'path':
            dataset = UMMCoTDatasetLoader._load_dataset_path(
                dataset_syntax.dataset,
                dataset_meta=dataset_meta,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                strict=strict,
                streaming=streaming,
                columns=columns,
                remove_unused_columns=remove_unused_columns,
            )
        else:
            subsets: List[SubsetDataset] = UMMCoTDatasetLoader._select_subsets(dataset_syntax.subsets, dataset_meta)
            revision = dataset_meta.hf_revision if use_hf else dataset_meta.ms_revision
            datasets = []
            for subset in subsets:
                dataset = UMMCoTDatasetLoader._load_repo_dataset(
                    dataset_syntax.dataset,
                    subset,
                    use_hf=use_hf,
                    hub_token=hub_token,
                    num_proc=num_proc,
                    load_from_cache_file=load_from_cache_file,
                    strict=strict,
                    revision=revision,
                    streaming=streaming,
                    download_mode=download_mode,
                    columns=columns,
                    remove_unused_columns=remove_unused_columns,
                )
                datasets.append(dataset)
            dataset = UMMCoTDatasetLoader._concat_datasets(datasets)
        return dataset


class UMMCOTPreprocessor(MessagesPreprocessor):
    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        processed_row = {}
        for branch_name, branch_input in row.items():
            processed_row[branch_name] = super().preprocess(branch_input)
        return processed_row
    
    @staticmethod
    def remove_useless_columns(dataset: DATASET_TYPE) -> DATASET_TYPE:
        dataset = UMMCOTPreprocessor.get_features_dataset(dataset)
        return dataset
    
    def batched_preprocess(self, batched_row: Dict[str, Any], *, strict: bool,
                           ignore_max_length_error: bool) -> Dict[str, Any]:
        from swift.llm.template import MaxLengthError
        batched_row = dict(batched_row)
        assert len(batched_row) > 0
        self._remove_prefix_keys(batched_row, '__@')  # compat streaming
        rows = self.batched_to_rows(batched_row)

        new_rows = []
        for row in rows:
            try:
                row = self.preprocess(row)
                for branch_name, branch_input in row.items():
                    for key in list(branch_input.keys()):
                        if key not in ['messages', 'images']:
                            del branch_input[key]
                # support [row1, row2, ...]
                if row is None:
                    row = []
                if isinstance(row, dict):
                    row = [row]
            except Exception as e:
                if strict:
                    logger.warning('To avoid errors, you can pass `strict=False`.')
                    raise
                if isinstance(e, MaxLengthError) and ignore_max_length_error:
                    pass
                elif self.traceback_limit is not None and self._traceback_counter < self.traceback_limit:
                    import traceback
                    logger.info(traceback.format_exc())
                    logger.warning('👆👆👆There are errors in the dataset, the data will be deleted')
                    self._traceback_counter += 1
                row = []
            new_rows += row
        res = self.rows_to_batched(new_rows)
        self._remove_prefix_keys(res, '__#')  # compat GRPO
        if len(res) == 0:
            res['messages'] = []
        return res


def load_dataset(
    datasets: Union[List[str], str],
    *,
    split_dataset_ratio: float = 0.,
    seed: Union[int, np.random.RandomState, None] = 42,
    num_proc: int = 1,
    load_from_cache_file: bool = True,
    shuffle: bool = False,
    streaming: bool = False,
    interleave_prob: Optional[List[float]] = None,
    stopping_strategy: Literal['first_exhausted', 'all_exhausted'] = 'first_exhausted',
    shuffle_buffer_size: int = 1000,
    use_hf: Optional[bool] = None,
    hub_token: Optional[str] = None,
    strict: bool = False,
    download_mode: Literal['force_redownload', 'reuse_dataset_if_exists'] = 'reuse_dataset_if_exists',
    columns: Optional[Dict[str, str]] = None,  # columns_mapping
    remove_unused_columns: bool = True,
    # self-cognition
    model_name: Optional[Union[Tuple[str, str], List[str]]] = None,  # zh, en
    model_author: Optional[Union[Tuple[str, str], List[str]]] = None,
) -> Tuple[DATASET_TYPE, Optional[DATASET_TYPE]]:
    """The interface to load any registered dataset

    Args:
        datasets: The dataset name list

        split_dataset_ratio: The dataset split ratio
        seed: The dataset random seed
        num_proc: Proc number to use when preprocess the dataset.
        shuffle: Whether to shuffle the dataset.
        streaming: Streaming mode or not
        use_hf: Use hf dataset or ms dataset.
        hub_token: The token of the hub.
        strict: Raise if any row is not correct.
        download_mode: Download mode, default is `reuse_dataset_if_exists`.
        columns: Used for manual column mapping of datasets.

        model_name: Model name in self-cognition task.
        model_author: Model author in self-cognition task
    Returns:
        The train dataset and val dataset
    """
    init_self_cognition_preprocessor(DATASET_MAPPING.get('self-cognition'), model_name, model_author)
    if isinstance(datasets, str):
        datasets = [datasets]
    if not isinstance(seed, np.random.RandomState):
        seed = np.random.RandomState(seed)
    if streaming:
        num_proc = None
    train_datasets = []
    val_datasets = []
    load_kwargs = {
        'num_proc': num_proc,
        'load_from_cache_file': load_from_cache_file,
        'strict': strict,
        'download_mode': download_mode,
        'columns': columns,
        'streaming': streaming,
        'hub_token': hub_token,
        'remove_unused_columns': remove_unused_columns,
    }
    use_hf_default = use_hf
    if use_hf_default is None:
        use_hf_default = True if use_hf_hub() else False
    for dataset in datasets:
        dataset_syntax = DatasetSyntax.parse(dataset)
        use_hf = dataset_syntax.use_hf or use_hf_default
        # compat dataset_name
        if dataset_syntax.dataset in DATASET_MAPPING:
            dataset_meta = DATASET_MAPPING[dataset_syntax.dataset]
            if dataset_syntax.use_hf is None and dataset_meta.dataset_path is not None:
                dataset_syntax.dataset = dataset_meta.dataset_path
                dataset_syntax.dataset_type = 'path'
            else:
                dataset_syntax.dataset = dataset_meta.hf_dataset_id if use_hf else dataset_meta.ms_dataset_id
        else:
            dataset_meta = dataset_syntax.get_dataset_meta(use_hf)
        # load_function = dataset_meta.load_function
        load_function = UMMCoTDatasetLoader.load
        train_dataset = load_function(dataset_syntax, dataset_meta, **load_kwargs, use_hf=use_hf)
        train_dataset, val_dataset = UMMCoTDatasetLoader.post_process(
            train_dataset,
            dataset_sample=dataset_syntax.dataset_sample,
            split_dataset_ratio=split_dataset_ratio,
            streaming=streaming,
            shuffle=shuffle,
            random_state=seed,
        )
        if train_dataset is not None:
            train_datasets.append(train_dataset)
        if val_dataset is not None:
            val_datasets.append(val_dataset)

    if interleave_prob is None:
        train_datasets = UMMCoTDatasetLoader._concat_datasets(train_datasets)
        val_datasets = UMMCoTDatasetLoader._concat_datasets(val_datasets)
    else:
        train_datasets = UMMCoTDatasetLoader._interleave_datasets(
            train_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)
        val_datasets = UMMCoTDatasetLoader._interleave_datasets(
            val_datasets, interleave_prob, seed=get_seed(seed), stopping_strategy=stopping_strategy)

    if shuffle:
        if train_datasets:
            train_datasets = UMMCoTDatasetLoader.shuffle_dataset(
                train_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
        if val_datasets:
            val_datasets = UMMCoTDatasetLoader.shuffle_dataset(
                val_datasets, seed=get_seed(seed), buffer_size=shuffle_buffer_size)
    return train_datasets, val_datasets


register_dataset(
DatasetMeta(
    dataset_name="my_training_data",
    dataset_path="data/json_files/ummcot_swift_his_20_val.jsonl",
    preprocess_func=UMMCOTPreprocessor(),
))

if __name__ == '__main__':
    # load_dataset returns train_dataset and val_dataset based on `split_dataset_ratio`
    # Here, since we didn't pass `split_dataset_ratio` (defaults to 0), we take the first one (index 0)
    dataset = load_dataset("my_training_data")[0]
    print(f'dataset[0]: {dataset[0]}')
