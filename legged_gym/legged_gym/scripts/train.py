import os
import numpy as np
from datetime import datetime
import sys

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args)
    
    # 保存配置文件的代码
    import shutil
    import inspect
    
    try:
        # 获取配置文件路径
        config_module = inspect.getmodule(env_cfg.__class__)
        config_file_path = config_module.__file__
        
        # 获取日志目录 - 根据RSL_RL框架调整
        log_dir = None
        if hasattr(ppo_runner, 'log_dir'):
            log_dir = ppo_runner.log_dir
        elif hasattr(ppo_runner, 'logger') and hasattr(ppo_runner.logger, 'log_dir'):
            log_dir = ppo_runner.logger.log_dir
        elif hasattr(ppo_runner, 'writer') and hasattr(ppo_runner.writer, 'log_dir'):
            log_dir = ppo_runner.writer.log_dir
        
        if log_dir:
            # 确保目录存在
            os.makedirs(log_dir, exist_ok=True)
            
            # 保存配置文件
            config_filename = os.path.basename(config_file_path)
            target_path = os.path.join(log_dir, config_filename)
            shutil.copy2(config_file_path, target_path)
            print(f"Config file saved to: {target_path}")
        else:
            print("Warning: Could not find log directory to save config file")
            
    except Exception as e:
        print(f"Warning: Failed to save config file: {e}")
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=train_cfg.runner.init_at_random_ep_len)

        
if __name__ == '__main__':
    args = get_args()
    train(args)
