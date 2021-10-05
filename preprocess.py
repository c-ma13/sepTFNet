import json
from pathlib import Path
from copy import deepcopy
import os

import soundfile as sf
from tqdm import tqdm

only_infer = 1
# reverb to direct separation
def json_gen(split_len=32000):
    data_base = Path("/data/machao/torch_separation_10db")
    # data_base = Path("/data/machao/datasets/librimix/Libri2Mix/wav8k/min")
    out_base = Path("/home/machao/torch_separation_freq/data/clean_8k_json")
    set_list = ["train", "valid", "infer"]
    if only_infer:
        set_list = ["infer"]

    for tmp_set in set_list:
        out_path_set = out_base / f"{tmp_set}_load.json"
        out_path_list = []

        tmp_path_set = data_base / tmp_set
        tmp_mix_path = tmp_path_set / 'mix'
        tmp_s1_path = tmp_path_set / 's1'
        tmp_s2_path = tmp_path_set / 's2'
        tmp_path_list = [os.path.basename(x) for x in list(tmp_mix_path.glob('*'))]

        if tmp_set in ["train", "valid"]:
            for tmp_file in tqdm(tmp_path_list):
                tmp_file_mix = tmp_mix_path / tmp_file
                tmp_file_s1 = tmp_s1_path / tmp_file
                tmp_file_s2 = tmp_s2_path / tmp_file

                tmp_samps, fs = sf.read(tmp_file_mix)
                tmp_len = len(tmp_samps)
                if tmp_len < split_len:
                    continue

                tmp_dict = {}
                tmp_dict["path_mix"] = os.path.abspath(tmp_file_mix)
                tmp_dict["path_s1"] = os.path.abspath(tmp_file_s1)
                tmp_dict["path_s2"] = os.path.abspath(tmp_file_s2)

                # tmp_dict["s1_name"] = tmp_pair_path[-2]
                # tmp_dict["s2_name"] = tmp_pair_path[-1]
                start = 0
                while (start + split_len < tmp_len):
                    end = start + split_len
                    tmp_split_dict = deepcopy(tmp_dict)
                    tmp_split_dict["start"] = start
                    tmp_split_dict["end"] = end
                    out_path_list.append(tmp_split_dict)
                    start = end
                if (tmp_len - start) > (split_len // 2):
                    start = tmp_len - split_len
                    end = tmp_len
                    tmp_split_dict = deepcopy(tmp_dict)
                    tmp_split_dict["start"] = start
                    tmp_split_dict["end"] = end
                    out_path_list.append(tmp_split_dict)
        elif tmp_set == "infer":
            for tmp_file in tqdm(tmp_path_list):
                tmp_file_mix = tmp_mix_path / tmp_file
                tmp_file_s1 = tmp_s1_path / tmp_file
                tmp_file_s2 = tmp_s2_path / tmp_file

                tmp_samps, fs = sf.read(tmp_file_mix)
                tmp_len = len(tmp_samps)

                tmp_dict = {}
                tmp_dict["path_mix"] = os.path.abspath(tmp_file_mix)
                tmp_dict["path_s1"] = os.path.abspath(tmp_file_s1)
                tmp_dict["path_s2"] = os.path.abspath(tmp_file_s2)
                tmp_dict["start"] = 0
                tmp_dict["end"] = tmp_len
                out_path_list.append(tmp_dict)

        out_path_set.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path_set, "w") as f:
            json.dump(out_path_list, f, indent=4)


if __name__ == "__main__":
    json_gen()
