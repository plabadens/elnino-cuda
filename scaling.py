import numpy as np
import pandas as pd
import re
import subprocess


def run_weak_scaling(
    min_blocks=1, max_blocks=32, samples=10, iterations=10000, threads=None
):
    output_pattern = r"elapsed time: (\d+\.\d+) s$"
    blocks = np.arange(min_blocks, max_blocks + 1)
    n_new = np.int32(30 * 40 * blocks)
    nx = np.int32(np.sqrt(n_new / (40 / 30)))
    ny = np.int32(n_new / nx)

    all_times = []
    base_args = ["./src/elnino_cuda", "--iter", str(iterations)]

    if not threads is None:
        base_args += ["--threads", str(threads), "--fperiod", str(iterations)]

    print("Starting single-threaded runs")
    for x, y, b in zip(nx, ny, blocks):
        args = base_args + ["--nx", str(x), "--ny", str(y), "--blocks", "1"]
        print(args)

        process = subprocess.Popen(args, stdout=subprocess.PIPE)

        output = process.communicate()[0].decode()

        match = re.search(output_pattern, output)

        if not match is None:
            output_value = float(match.group(1))
            all_times.append(output_value)

    print("Starting scaling runs")
    for x, y, b in zip(nx, ny, blocks):
        args = base_args + ["--nx", str(x), "--ny", str(y), "--blocks", str(b)]
        print(args)

        for _ in range(samples):
            process = subprocess.Popen(args, stdout=subprocess.PIPE)

            output = process.communicate()[0].decode()

            match = re.search(output_pattern, output)

            if not match is None:
                output_value = float(match.group(1))
                all_times.append(output_value)

    df = pd.DataFrame(
        {
            "blocks": np.concatenate((np.repeat(1, len(blocks)), np.repeat(blocks, samples))),
            "iter": np.repeat(iterations, len(blocks) * samples + len(blocks)),
            "nx": np.concatenate((nx, np.repeat(nx, samples))),
            "ny": np.concatenate((nx, np.repeat(ny, samples))),
            "time": all_times,
        }
    )

    return df


if __name__ == "__main__":
    data = run_weak_scaling(min_blocks=1, max_blocks=32, iterations=10000, threads=1)
    data.to_csv("weak_scaling.csv", index=False)
