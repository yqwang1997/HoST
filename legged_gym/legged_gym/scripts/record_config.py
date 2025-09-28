import os
from legged_gym import LEGGED_GYM_ENVS_DIR

def record_config(log_root, name):
    log_dir=log_root
    os.makedirs(log_dir, exist_ok=True)

    str_config = name + '_config.txt'
    file_path1=os.path.join(log_dir, str_config)
    file_path2=os.path.join(log_dir, 'legged_robot_config.txt')
    
    root1 = name.split('_')[0]
    if name == "CAS02_ground_19dof":
        root_path1 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, 'CAS02_config_ground_19dof.py')
        root_path2 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'host_ground_19dof.py')
    elif name == "CAS02_ground":
        root_path1 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, 'CAS02_config_ground.py')
        #root_path1 = os.path.join('/home/casbot/ZHZ_ws/HoST/legged_gym/legged_gym/envs/CAS02/CAS02_config_ground.py')
        root_path2 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'host_ground.py')
    elif name == "CAS02_ground_prone":
        root_path1 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, 'CAS02_config_ground_prone.py')
        root_path2 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'host_ground_prone.py')
    elif name == "CAS02_ground_HiFar":
        root_path1 = os.path.join(LEGGED_GYM_ENVS_DIR, root1, 'CAS02_config_ground_HiFar.py')
        root_path2 = os.path.join(LEGGED_GYM_ENVS_DIR, 'base', 'host_ground_HiFar.py')



    with open(root_path1, 'r', encoding='utf-8') as file:
        content = file.read()

    with open(file_path1, 'w', encoding='utf-8') as file:
        file.write(content)

    with open(root_path2, 'r',encoding='utf-8') as file:
        content = file.read()

    with open(file_path2, 'w', encoding='utf-8') as file:
        file.write(content)