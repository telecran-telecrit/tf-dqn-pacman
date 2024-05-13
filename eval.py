from evaluation_old import *
from evaluation_old.parser_old import args
from pathlib import Path
import re

NUM = re.compile(r"episode-(\d+).pkl")
def dir_path(path):
    if path == "" or Path(path).exists():
        return path
    else:
        raise NotADirectoryError(path)

path = dir_path(args.path)
ABS_PATH = Path().absolute()
if path == "":
    RESULTS_PATH = ABS_PATH / "results"
    recent = lambda folder: folder.stat().st_mtime
    WORKING_DIRECTORY = max(RESULTS_PATH.iterdir(), key=recent)
else:
    WORKING_DIRECTORY = Path(path)

if args.episode == "last":
    get_num = lambda s: int(NUM.search(str(s))[1])
    selected_episode = max(map(get_num, (WORKING_DIRECTORY / "recorded-data").iterdir()))
else:
    selected_episode = args.episode

print('Opening data of episode {} in "{}"'.format(selected_episode, WORKING_DIRECTORY))

if args.all:
    record(selected_episode, WORKING_DIRECTORY)
    only_rewards(selected_episode, WORKING_DIRECTORY)
    only_q_values(selected_episode, WORKING_DIRECTORY)
    load_save_result(selected_episode, WORKING_DIRECTORY)
else:
    if args.record:
        record(selected_episode, WORKING_DIRECTORY)
    elif not (args.record):
        load_save_result(selected_episode, WORKING_DIRECTORY)
    if args.reward:
        only_rewards(selected_episode, WORKING_DIRECTORY)
    if args.qvalue:
        only_q_values(selected_episode, WORKING_DIRECTORY)
