from habitat_base.simulation import SceneSimulator
from .common_utils import transform_position, transform_rotation, transback_rotation, rotation_matrix_to_euler_angles, euler_angles_to_rotation_matrix
import numpy as np
import math
import torch
import time
import os
from NavModel.VLMModel import My_VLM_NavModel
from PIL import Image
import json

label_index = {
    "stop": np.array([0]), # stop
    "turn_left": np.array([1]),
    "move_forward": np.array([2]),
    "turn_right": np.array([3]),
}


def check_checkpoint(args, model, optimizer, lr_scheduler) -> int:
    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model_state_dict = model.state_dict()
        state_disk = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        update_model_state = {}
        for key, val in state_disk.items():
            if key in model_state_dict and model_state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print(
                    'Ignore weight %s: %s' % (key, str(val.shape))
                )
        msg = model.load_state_dict(update_model_state, strict=False)

        if 'epoch' in checkpoint:
            resume_from_epoch = checkpoint['epoch'] + 1
            print("Resume from Epoch {}".format(resume_from_epoch))
            optimizer.load_state_dict(checkpoint['optimizer'])

    return resume_from_epoch

def save_checkpoint(model, model_path, optimizer=None, epoch: int=0, save_states: bool=False):
    if hasattr(model, 'module'):
        model = model.module
    
    state_dict = {
        "model_state_dict": model.state_dict()
    }
    if save_states:
        state_dict.update({
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        })

    torch.save(state_dict, model_path)
    
    
# base training agent class for Habitat
# it contains the training and validation loop
class HabitatAgent:
    def __init__(self, args, config, nav_model, subset_name=None):
        self.args = args
        self.config = config
        self.nav_model = nav_model
        if not isinstance(self.nav_model, My_VLM_NavModel):
            self.nav_model.reset_episode(1)

        self.task_sim = SceneSimulator(args=args, config=config)

        self.subset_name = subset_name


    def validate(self, step_task=False):  
        # 获取当前环境信息
        action = 'stop'
        obs, done, info = self.task_sim.actor(action)
        
        self.task_sim.gt_step = self.task_sim.count_gt_step(step_task)
        if step_task:
            pos = self.config["start_pos"]
            yaw = math.degrees(self.config["start_yaw"] - 180)
            rot = transback_rotation(euler_angles_to_rotation_matrix(np.array([0, 0, yaw])))
            self.task_sim.set_state(pos, rot)
        
        if isinstance(self.nav_model, My_VLM_NavModel):
            self.nav_model.his_images = []
            action_list = []  # multi step prediction model
        else:
            action_list = None
        
        img_idx = 0
        stop_count = 0
        if self.subset_name is None:
            cache_dir = ".cache"
        else:
            cache_dir = f".cache/{self.subset_name}"
        os.makedirs(cache_dir, exist_ok=True)


        navigation_videos = {}
        his_actions = {}

        previous_stage = 0

        time_start = time.time() # Debug: Time taken for one episode

        while not(self.task_sim.episode_over):
            agent_pos = transform_position(info["agent position"])
            agent_rot = transform_rotation(info["agent rotation"])

            if previous_stage != self.task_sim.stage:  # new stage
                previous_stage = self.task_sim.stage

            obs_path = []
            for img in obs:
                out_path = os.path.join(cache_dir, f"{img_idx}.png")
                img.save(out_path)
                obs_path.append(out_path)
                img_idx += 1

            if isinstance(self.nav_model, My_VLM_NavModel):
                view_inputs = obs_path  # VLM 直接传图像路径

            input = {
                'observations': [
                    {
                        "instruction": self.task_sim.ins,                                       # instruction
                        "view_feats": view_inputs,                                 # "view features"
                        "pose": np.append(agent_pos, rotation_matrix_to_euler_angles(agent_rot)[2]),    # xyz,heading
                        "stop_count": stop_count
                    }
                ],
            }

            print(f"Debug: {self.task_sim.ins}")

            label = self.task_sim.get_next_action(info["target coord"])
            if action_list is None:    
                action, output = self.nav_model.step(input["observations"])
            else:
                assert isinstance(action_list, list)
                if len(action_list) == 0:
                    action_list, output = self.nav_model.step(input["observations"])
                
                if len(action_list) == 0:  # BUG: Model failed to generate action and then stop task
                    print("Model failed to generate action and then stop task")
                    action = 'stop'
                else:
                    action = action_list.pop(0)
            
            # import pdb; pdb.set_trace()

            # print gt action
            print(f"The ground truth action for step {self.task_sim.step} is {label}")                    
            
            # Record historical frames
            if self.task_sim.stage not in navigation_videos:
                navigation_videos[self.task_sim.stage] = []
            navigation_videos[self.task_sim.stage].append(obs_path[1])
            # Record historical actions
            if self.task_sim.stage not in his_actions:
                his_actions[self.task_sim.stage] = []
            his_actions[self.task_sim.stage].append(action)

            obs, done, info = self.task_sim.actor(action)

            # Update historical visual observations
            self.nav_model.his_images.append(obs_path[1])
            if len(self.nav_model.his_images) > 20:
                self.nav_model.his_images = self.nav_model.his_images[-20:]
            print(len(self.nav_model.his_images))

            if action == 'stop':
                self.nav_model.his_images = []
            
            print("success: ", self.task_sim.successes)
            print("stage: ", self.task_sim.stage)

        print(f"Time taken: {time.time() - time_start} seconds")  # Debug: Time taken for one episode
        self.task_sim.close()

        stage = 0
        succ_root = "outputs/success_traj"
        for key, value in navigation_videos.items():
            print(stage)
            print(navigation_videos.keys())
            if self.task_sim.successes[stage]:
                os.makedirs(succ_root, exist_ok=True)

                traj_dir = os.path.join(succ_root, self.task_sim.ins, f"{stage+1}")
                os.makedirs(traj_dir, exist_ok=True)

                img_idx = 0
                for img_path in value:
                    img = Image.open(img_path)
                    out_path = os.path.join(traj_dir, f"{img_idx}.png")
                    img.save(out_path)
                    img_idx += 1
            stage += 1

        if any(self.task_sim.successes):
            action_dir = os.path.join(succ_root, self.task_sim.ins, "actions.json")
            with open(action_dir, "w", encoding="utf-8") as f:
                json.dump(his_actions, f, ensure_ascii=False, indent=2)

        return self.task_sim.return_results()
