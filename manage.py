# move all folders and files under /experiments to /experiments/.old
import os
import shutil
import config
from datetime import datetime
from argparse import ArgumentParser

# parse arguments
parser = ArgumentParser()
subparsers = parser.add_subparsers(title='Commands', dest='cmd')
ls_p = subparsers.add_parser('ls', help='List experiments')
ls_p.add_argument('-a', '--all', action='store_true', help='List all experiments including those under .old/.')
cl_p = subparsers.add_parser('clean', help='Clean all experiments. move all folders and files to .old/.')
# parser.add_argument('-cl', '--clean', action='store_true', help='clean all experiments. move all folders and files to .old/.')
p_p = subparsers.add_parser('purge', help='Purge all experiments under .old/.')
args = parser.parse_args()

# list all experiments
if args.cmd == 'ls':
    # get all folders under /experiments
    experiments = os.listdir(config.experiments_folder)

    # print all experiments
    for experiment in experiments:
        if experiment != '.old':
            print(experiment)
    if args.all:
        experiments_old = os.listdir(os.path.join(config.experiments_folder, '.old'))
        for experiment in experiments_old:
            print(experiment)
    exit(0)

# clean all experiments
if args.cmd == 'clean':
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
            shutil.move(os.path.join(args.experiments_folder, experiment), 
                        os.path.join(args.experiments_folder, '.old', experiment))
            
    # print summary
    print(f"Moved {num_experiments} experiments to {os.path.join(config.experiments_folder, '.old')}")
    exit(0)

# purge all experiments
if args.cmd == 'purge':
    input = input("Are you sure to purge all experiments under .old/? (y/n)")
    if input != 'y':
        exit(0)

    # get all folders under /experiments/.old
    experiments = os.listdir(os.path.join(config.experiments_folder, '.old'))

    # remove all folders and files under /experiments/.old
    for experiment in experiments:
        shutil.rmtree(os.path.join(config.experiments_folder, '.old', experiment))
    
    # print summary
    print(f"Purged {len(experiments)} experiments from {os.path.join(config.experiments_folder, '.old')}")
    exit(0)