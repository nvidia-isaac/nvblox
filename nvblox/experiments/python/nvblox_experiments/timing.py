import re
import pandas as pd

def get_timings_as_dataframe(filepath: str):
    names = []
    num_calls = []
    total_times = []
    means = []
    stds = []
    mins = []
    maxes = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:
            entries = re.split('\s+|,', line)
            entries = [entry.strip('()[]') for entry in entries]

            names.append(entries[0])
            num_calls.append(int(entries[1]))
            total_times.append(float(entries[2]))
            means.append(float(entries[3]))
            stds.append(float(entries[5]))
            mins.append(float(entries[6]))
            maxes.append(float(entries[7]))

    d = {"num_calls": num_calls, "total_time": total_times,
         "mean": means, "std": stds, "min": mins, "max": maxes}
    timings = pd.DataFrame(d, index=names)
    return timings
