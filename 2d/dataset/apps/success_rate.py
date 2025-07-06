import os
import numpy as np
import argparse

def smoke_rate(datapath, data_begin, data_end):
    sim_smoke_rate_list = []
    smoke_rate_sum = 0

    for i in range(data_begin, data_end):
        if i < 10:
            sim_name = f'sim_00000{i}'
        elif i < 100:
            sim_name = f'sim_0000{i}'
        elif i < 1000:
            sim_name = f'sim_000{i}'
        elif i < 10000:
            sim_name = f'sim_00{i}'
        elif i < 100000:
            sim_name = f'sim_0{i}'
        sim_path = os.path.join(datapath, sim_name)
        smoke_path = os.path.join(sim_path, "Smoke.npy")
        if not os.path.exists(smoke_path):
            continue
        smoke_data = np.load(smoke_path)
        smoke_target = smoke_data[-1][1]
        smoke_sum = np.sum(smoke_data[-1])
        smoke_rate = smoke_target / smoke_sum
        smoke_rate_sum += smoke_rate
        sim_smoke_rate_list.append(smoke_rate)
        if i % 100==0:
            print(f"{sim_name} down!")

    smoke_rate_data = smoke_rate_sum/(data_end-data_begin)

    print(f'smoke_rate_data: {smoke_rate_data}')

    x_values = list(range(data_begin, data_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="data path")
    parser.add_argument("--data_begin", type=str, help="begin index of data")
    parser.add_argument("--data_end", type=str, help="end index of data")

    args = parser.parse_args()
    datapath = args.datapath
    data_begin = int(args.data_begin)
    data_end = int(args.data_end)
    smoke_rate(datapath=datapath, data_begin=data_begin, data_end=data_end)
