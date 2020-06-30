import pandas as pd
import os

csv_path = 'data/newsflare-2/Newsflare-361552-unlikely-friends-sheep-and-dog_merge.csv'
assert os.path.isfile(csv_path)
save_path = csv_path[:-10] + '.txt'
csv_read = pd.read_csv(csv_path)


def get_set(item_csv):
    item_list = []
    for item in item_csv:
        if not pd.isna(item):
            item = item.strip()
            item_list += item.split(', ')
    return set(item_list)


obj_set = get_set(csv_read['Objects'])
act_set = get_set(csv_read['Actions'])
sce_set = get_set(csv_read['Scene'])

with open(save_path, 'w') as f:
    f.write(csv_path[:-10] + '\n\n')
    f.write('Objects:\n')
    f.write((', ').join(obj_set))
    f.write('\n\nActions:\n')
    f.write((', ').join(act_set))
    f.write('\n\nScene:\n')
    f.write((', ').join(sce_set))

