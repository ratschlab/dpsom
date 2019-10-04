""" 
I/O utilities
"""

import csv
import pickle

def print_list(lst):
    for elem in lst:
        print(str(elem))

def load_pickle(fpath):
    with open(fpath, 'rb') as fp:
        return pickle.load(fp)

def save_pickle(obj, out_fpath):
    with open(out_fpath, 'wb') as fp:
        pickle.dump(obj, fp)

def write_list_to_file(fpath, lst):
    with open(fpath, 'w') as fp:
        for elem in lst:
            print(str(elem).strip(), file=fp)

def read_list_from_file(fpath, skip_header=False):
    with open(fpath, 'r') as fp:
        lines = fp.read().splitlines()
        if skip_header:
            return lines[1:]
        else:
            return lines

def dict_from_text_file(fpath):
    out_dict = {}

    with open(fpath, "r") as fp:
        for line in fp:
            comps = list(map(lambda elem: elem.strip(), line.strip().split(",")))
            assert(len(comps) == 2)
            key = comps[0]
            val = comps[1]
            out_dict[key] = float(val)

    return out_dict

def read_dict_iter(filename):
    with open(filename) as thefile:
        reader = csv.DictReader(thefile, delimiter='\t')
        for datum in reader:
            yield datum

def dict_to_text_file(fpath, in_dict):
    with open(fpath, "w") as fp:
        for key in in_dict:
            print("{},{:.5f}".format(key.strip(), in_dict[key]), file=fp)
