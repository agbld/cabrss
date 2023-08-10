# move all folders and files under /experiments to /experiments/.old
import os
import shutil
import config
from datetime import datetime

# get all folders under /experiments
experiments = os.listdir(config.experiments_folder)

# make /experiments/.old if not exist
if '.old' not in experiments:
    os.mkdir(os.path.join(config.experiments_folder, '.old'))
    num_experiments = len(experiments)
else:
    num_experiments = len(experiments) - 1

# move all folders and files under /experiments to /experiments/.old
for experiment in experiments:
    if experiment != '.old':
        # get datetime string
        datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")

        # move and add a postfix to the folder name by datetime
        shutil.move(os.path.join(config.experiments_folder, experiment), 
                    os.path.join(config.experiments_folder, '.old', experiment + '.' + datetime_str))
# print summary
print(f"Moved {num_experiments} experiments to {os.path.join(config.experiments_folder, '.old')}")
