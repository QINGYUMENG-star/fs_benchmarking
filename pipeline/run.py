import argparse, os, time
parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="inp", required=True)
parser.add_argument("--out", dest="out", required=True)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
time.sleep(5)
with open(os.path.join(args.out, "report.txt"), "w") as f:
    f.write(f"Processed: {args.inp}\nStatus: OK\n")
print("done")
