from typing import List, Dict, Any, Optional

from swift.llm.template.template.qwen import Qwen2_5VLTemplate


class MyQwen2_5VLTemplate(Qwen2_5VLTemplate):
    my_qwen2_5_vl = 'my_qwen2_5_vl'

    def encode(
        self,
        inputs: Dict[str, Any],
        return_template_inputs: bool = False,
        return_length: bool = False
    ) -> Dict[str, Any]:
        encoded_inputs = {}
        for branch_name, branch_input in inputs.items():
            encoded_inputs[branch_name] = super().encode(
                branch_input,
                return_template_inputs=return_template_inputs,
                return_length=return_length
            )
        return encoded_inputs
    
    def _data_collator(
        self,
        batch: List[Dict[str, Any]],
        *,
        padding_to: Optional[int] = None
    ) -> Dict[str, Any]:
        collated_batch = {}
        for branch_name in batch[0].keys():
            branch_batch = []
            for sample in batch:
                branch_batch.append(sample[branch_name])
            collated_batch[branch_name] = super()._data_collator(branch_batch, padding_to=padding_to)
        return collated_batch
