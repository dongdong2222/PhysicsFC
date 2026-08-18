from pathlib import Path
import yaml
import pprint

# motion data list write
# npy file needed
# all files should be this directory

directory = Path('./')
config_file = Path('./soccer_9.yaml')
motion_pathes = list()
config = dict()

for path in directory.iterdir():
    if 'calm.npy' in str(path):
        motion_pathes.append(path)

with open(str(config_file), 'w') as f:
    config['motions'] = [{'file': str(motion_path), 'weight': 0.2} for motion_path in motion_pathes]
    yaml.dump(config, f)