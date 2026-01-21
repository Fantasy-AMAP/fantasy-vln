import re
import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from torch.nn.parallel import DistributedDataParallel as DDP

import random
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def extract_substr(input_str, start_token="<answer>", end_token="</answer>"):
    # import pdb; pdb.set_trace()
    start_token = re.escape(start_token)
    end_token = re.escape(end_token)
    pattern = f"{start_token}(.*?){end_token}"
    substr = re.findall(pattern, input_str, re.DOTALL)
    # print(f"Debug: {input_str}")
    return substr


class My_VLM_NavModel:
    def __init__(self, model_id, device, use_ddp=False):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(device)

        input_emb = model.get_input_embeddings()
        embed_weights = input_emb.weight

        new_lm_head_weight = torch.zeros_like(embed_weights)
        model.lm_head.weight = nn.Parameter(new_lm_head_weight)

        idx_file = 'model.safetensors.index.json'
        idx_path = os.path.join(model_id, idx_file)
        idx_data = json.load(open(idx_path))
        lm_head_mapping = idx_data['weight_map']['lm_head.weight']
        lm_head_ckpt = os.path.join(model_id, lm_head_mapping)
        state_dict = load_file(lm_head_ckpt)
        for key in list(state_dict.keys()):
            if 'lm_head' in key:
                new_key = key[len('lm_head.'):]
                state_dict[new_key] = state_dict.pop(key)
        missing_keys, unexpected_keys = model.lm_head.load_state_dict(state_dict, strict=False)
        assert len(missing_keys) == 0
        del state_dict

        if use_ddp:
            self.model = DDP(
                model,
                device_ids=[device],
                output_device=device,
                find_unused_parameters=False
            )
        else:
            self.model = model

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.his_images = []
        self.model_name = "FantasyVLN"

    def _get_prompt(self, task_ins, his_img_num, stop_count, model_name="FantasyVLN"):
        if model_name == "FantasyVLN":
            prefix = (
                "<no_textual_think><no_visual_think>"
                "You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.\n"
                "Based on these information, you need to decide your next 5 actions, which could involve <|left|>,<|right|>,<|forward|>. "
                "If you finish your mission, output <|stop|>. "
                "Here are some examples: <|left|><|forward|><|forward|><|stop|>, "
                "<|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>\n"
            )
            history_part = ""
            if his_img_num > 0:
                history_imgs = "".join(["<image>" for _ in range(his_img_num - 3)])
                history_part = f"# Your historical pictures are: {history_imgs}\n"

            current_part = (
                "# Your current observations is leftside: <image>, frontside: <image>, rightside: <image>\n"
            )
            
            mission_part = f"# Your mission is: {task_ins}"
            note_part = (
                "PS: The mission is complex. You may infer several sub-tasks within the mission, and output <|stop|> when a sub-task is achieved. "
                f"So far, you have output <|stop|> {stop_count} times. Historical information reflects progress up to the current subgoal. <|NAV|>"
            )
            return prefix + history_part + current_part + mission_part
        elif model_name == "WorldVLN":
            return f"What action should the robot take to {task_ins}?\n Your current observations are left side: <image>, front side: <image>, right side: <image>"
        else:
            raise NotImplementedError
    
    def _prepare_nav_inputs(self, prompts, his_imgs, obs_imgs):
        nav_inputs = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text", 
                                    "text": prompts
                                },
                            ],
                        }
                    ]

        imgs = his_imgs + obs_imgs
        for img in imgs:
            nav_inputs[0]['content'].append(
                {
                    "type": "image",
                    "image": img
                }
            )

        return nav_inputs
    
    def _get_action(self, nav_outs):
        if self.model_name == "FantasyVLN":
            action = extract_substr(nav_outs)
            if action == []:
                action = extract_substr(nav_outs, '<var>', '</answer>')
                if action == []:
                    action = random.choice(['<|forward|>', '<|left|>', '<|right|>', '<|stop|>'])
                else:
                    action = action[0]
            else:
                action = action[0]
            
            try:
                assert isinstance(action, str)
            except:
                action = random.choice(['<|forward|>', '<|left|>', '<|right|>', '<|stop|>'])
                print("Debug: ", action, " ", type(action))
        elif self.model_name == "WorldVLN":
            action = nav_outs
        else:
            raise NotImplementedError
        
        action = extract_substr(action, '<|', '|>')
        action = [f"<|{act}|>" for act in action]
        nav_acts = []

        for act in action:
            if act == '<|forward|>':
                act = 'move_forward'
                nav_acts.append(act)
            elif act == '<|left|>':
                act = 'turn_left'
                nav_acts.append(act)
            elif act == '<|right|>':
                act = 'turn_right'
                nav_acts.append(act)
            elif act == '<|stop|>':
                act = 'stop'
                nav_acts.append(act)
            else:
                print(f"--warning-- Illigal action: {act}, \nModel outputs: {nav_outs}")
                pass
        
        return nav_acts

    @torch.no_grad()
    def step(self, batch_inputs):
        # import pdb; pdb.set_trace()
        # prepare inputs
        task_ins = batch_inputs[0]['instruction']
        obs_imgs = batch_inputs[0]['view_feats']

        if self.model_name == "FantasyVLN":
            stop_count = batch_inputs[0]['stop_count']
        elif self.model_name == "WorldVLN":
            stop_count = None
        else:
            raise NotImplementedError

        nav_prompt = self._get_prompt(task_ins, len(self.his_images) + len(obs_imgs), stop_count, self.model_name)
        if self.model_name == "FantasyVLN":
            nav_inputs = self._prepare_nav_inputs(nav_prompt, self.his_images, obs_imgs)
        elif self.model_name == "WorldVLN":
            nav_inputs = self._prepare_nav_inputs(nav_prompt, [], obs_imgs)
        else:
            raise NotImplementedError
        
        # import pdb; pdb.set_trace()

        # model forward
        text = self.processor.apply_chat_template(
            nav_inputs, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(nav_inputs)

        new_template_text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
        new_template_text = new_template_text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        text = new_template_text

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        nav_outs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # get actions
        action = self._get_action(nav_outs[0])

        # self.his_images.extend(obs_imgs)

        # import pdb; pdb.set_trace()
        print("input img num", len(self.his_images) + len(obs_imgs))
        
        return action, None
