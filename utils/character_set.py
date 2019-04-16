import sys
import json
import os
from collections import defaultdict

def load_char_set(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k,v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v

    return idx_to_char, char_set['char_to_idx']

if __name__ == "__main__":
    character_set_path = sys.argv[-1]
    out_char_to_idx = {}
    out_idx_to_char = {}
    char_freq = defaultdict(int)
    cnt=1
    for i in range(1, len(sys.argv)-1):
        data_file = sys.argv[i]
        if data_file[-4:]=='.txt':
            with open(data_file) as f:
                text = f.read()
            for c in text:
                if c not in out_char_to_idx and c!='\n':
                    out_char_to_idx[c] = cnt
                    out_idx_to_char[cnt] = c
                    cnt += 1
                char_freq[c] += 1
        elif data_file[-4:]=='json':
            print(data_file)
            with open(data_file) as f:
                paths = json.load(f)

            for json_path, image_path in paths:
                with open(json_path) as f:
                    data = json.load(f)

                cnt = 1 # this is important that this starts at 1 not 0
                for data_item in data:
                    for c in data_item.get('gt', None):
                        if c is None:
                            print("There was a None GT")
                            continue
                        if c not in out_char_to_idx:
                            out_char_to_idx[c] = cnt
                            out_idx_to_char[cnt] = c
                            cnt += 1
                        char_freq[c] += 1


    out_char_to_idx2 = {}
    out_idx_to_char2 = {}

    remove_thresh=500

    removed=0
    for i, c in enumerate(sorted(out_char_to_idx.keys())):
        if char_freq[c]>remove_thresh:
            out_char_to_idx2[c] = i+1-removed
            out_idx_to_char2[i+1-removed] = c
        else:
            removed+=1

    for k,v in sorted(iter(char_freq.items()), key=lambda x: x[1]):
        if v<=remove_thresh:
            print('removed:',k, v)
        else:
            print(k, v)


    output_data = {
        "char_to_idx": out_char_to_idx2,
        "idx_to_char": out_idx_to_char2
    }


    print(("Size:", len(output_data['char_to_idx'])))

    with open(character_set_path, 'w') as outfile:
        json.dump(output_data, outfile)
