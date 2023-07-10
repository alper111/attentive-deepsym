import time
import os
import argparse
import subprocess
from multiprocessing import Process


def collect(script, num, t, folder, idx):
    subprocess.run(["python", script, "-N", num, "-T", t, "-o", folder, "-i", idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Collect interaction data in parallel.")
    parser.add_argument("-s", help="script", type=str, required=True)
    parser.add_argument("-d", help="data folder", type=str, required=True)
    parser.add_argument("-N", help="number of data per proc", type=int, required=True)
    parser.add_argument("-T", help="interaction per episode", type=int, required=True)
    parser.add_argument("-p", help="number of procs", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.d):
        os.makedirs(args.d)

    procs = []
    start = time.time()
    for i in range(args.p):
        p = Process(target=collect, args=[args.s, str(args.N), str(args.T), args.d, str(i)])
        p.start()
        procs.append(p)

    for i in range(args.p):
        procs[i].join()
    end = time.time()
    elapsed = end - start
    print(args.N, file=open(os.path.join(args.d, "info.txt"), "w"))
    print(args.p, file=open(os.path.join(args.d, "info.txt"), "a"))
    print(f"Collected {args.p*args.N} samples in {elapsed:.2f} seconds. {args.p*args.N/elapsed}")
