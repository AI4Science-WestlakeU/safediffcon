import os
import numpy as np
import argparse

def smoke_rate(datapath, data_begin, data_end, c_bound):
    sim_smoke_rate_list1 = []
    sim_smoke_rate_list2 = []
    sim_smoke_rate_list3 = []
    smoke_rate_sum1 = 0
    smoke_rate_sum2 = 0
    smoke_rate_sum3 = 0
    unsafe_count = 0

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
        smoke_safe_path = os.path.join(sim_path, "Smoke_safe.npy")
        if not os.path.exists(smoke_safe_path):
            continue
        smoke_safe_data = np.load(smoke_safe_path)
        # smoke_data = np.load(smoke_path)
        smoke_target1 = smoke_safe_data[-1][0]
        smoke_target2 = smoke_safe_data[-1][1]
        smoke_target3 = smoke_safe_data[-1][2]
        smoke_sum = np.sum(smoke_safe_data[-1])
        # print(smoke_sum)
        smoke_rate_sum1 += smoke_target1 / smoke_sum
        smoke_rate_sum2 += smoke_target2 / smoke_sum
        smoke_rate_sum3 += smoke_target3 / smoke_sum
        sim_smoke_rate_list1.append(smoke_rate_sum1)
        sim_smoke_rate_list2.append(smoke_rate_sum2)
        sim_smoke_rate_list3.append(smoke_rate_sum3)
        if smoke_target1 / smoke_sum > c_bound:
            unsafe_count += 1
        if i % 100==0:
            print(f"{sim_name} down!")

    smoke_rate_data1 = smoke_rate_sum1/(data_end-data_begin)
    smoke_rate_data2 = smoke_rate_sum2/(data_end-data_begin)
    smoke_rate_data3 = smoke_rate_sum3/(data_end-data_begin)

    print(f'smoke_safe_rate_data1: {smoke_rate_data1}')
    print(f'smoke_safe_rate_data2: {smoke_rate_data2}')
    print(f'smoke_safe_rate_data3: {smoke_rate_data3}')
    print(f'unsafe_count: {unsafe_count}')
    print(f'unsafe_rate: {unsafe_count/(data_end-data_begin)}')

    x_values = list(range(data_begin, data_end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="data path")
    parser.add_argument("--data_begin", type=str, help="begin index of data")
    parser.add_argument("--data_end", type=str, help="end index of data")
    parser.add_argument("--c_bound", type=float, help="bound of safety score c")

    args = parser.parse_args()
    datapath = args.datapath
    data_begin = int(args.data_begin)
    data_end = int(args.data_end)
    smoke_rate(datapath=datapath, data_begin=data_begin, data_end=data_end, c_bound=args.c_bound)
