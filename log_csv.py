import pandas as pd
import os
import argparse


def parse_log(log, getTime = False):
    time = []
    tags = []
    for line in log:
        cur_time, cur_tags = line.split('\t')
        if getTime:
            time.append(cur_time.strip())
        tags.append(cur_tags.strip())
    if getTime:
        return tags, time
    return tags


parser = argparse.ArgumentParser()
#parser.add_argument('--action', default=None, help='path of action log')
#parser.add_argument('--object', default=None, help='path of object log')
#parser.add_argument('--scene', default=None, help='path of scene log')
parser.add_argument('--video', default=None, help='path of video')
args = parser.parse_args()

act_path = os.path.splitext(args.video)[0] + '_act.log'
obj_path = os.path.splitext(args.video)[0] + '_objAll.log'
scene_path = os.path.splitext(args.video)[0] + '_scene.log'
assert os.path.isfile(act_path), 'Wrong path of action log:%s' % act_path
assert os.path.isfile(obj_path), 'Wrong path of object log:%s' % obj_path
assert os.path.isfile(scene_path), 'Wrong path of scene log:%s' % scene_path

act_logs = open(act_path, 'r').readlines()
obj_logs = open(obj_path, 'r').readlines()
scene_logs = open(scene_path, 'r').readlines()

assert len(act_logs) == len(obj_logs) == len(scene_logs), 'three logs are not in equal length'

save_path = os.path.splitext(act_path)[0][:-3] + 'merge.csv'

actions, timestamps = parse_log(act_logs, getTime=True)
objects = parse_log(obj_logs)
scenes = parse_log(scene_logs)

data = {'Timestamp': timestamps, 'Objects': objects, 'Actions': actions, 'Scene': scenes}
df = pd.DataFrame(data)
df.to_csv(save_path, index=False)
