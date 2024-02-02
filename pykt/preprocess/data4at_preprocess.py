import pandas as pd
from .utils import sta_infos, write_txt, format_list2str, change2timestamp, replace_text
import json
import os

KEYS = ["stu_id", "questions", "step_kc"]


def read_data_from_csv(read_file, write_file, dataset_name=None):
    if not dataset_name is None:
        write_file = write_file.replace("/data4at_1212/", f"/{dataset_name}/")
        write_dir = read_file.replace("/data4at_1212/", f"/{dataset_name}/")
        print(f"write_dir is {write_dir}")
        print(f"write_file is {write_file}")
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

    stares = []
    df = pd.read_csv(os.path.join(read_file, f"stu_data_1219.csv"), low_memory=False)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    df = df.dropna(subset=["stu_id", "questions", "step_kc", "responses", "log_idx"])
    print(f"df1:{df.shape}")

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(
        f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}"
    )

    user_inters = []
    cnt = 0
    for ui in df.iterrows():
        user, tmp_inter = ui[0], ui[1]
        seq_skills = tmp_inter["step_kc"].split(",")
        seq_ans = tmp_inter["responses"].split(",")
        seq_problems = tmp_inter["questions"].split(",")
        seq_start_time = tmp_inter["log_idx"].split(",")
        seq_response_cost = ["NA"]

        assert len(seq_problems) == len(seq_skills) == len(seq_ans)

        user_inters.append(
            [
                [str(user), str(len(seq_problems))],
                seq_problems,
                seq_skills,
                seq_ans,
                seq_start_time,
                seq_response_cost,
            ]
        )
        cnt += 1
        if dataset_name == "data4at_debug" and cnt == 10000:
            break
        if dataset_name == "data4at_1212_1w" and cnt == 10000:
            break
        if dataset_name == "data4at_1212_5w" and cnt == 50000:
            break

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return write_dir, write_file
