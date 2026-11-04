"""Pilot for the measured_axis fiber layout: render the same rows with both layouts.

python dev/pilot_measured_axis.py --rows 0 1 2 --tag dev_fiberpilot
"""
import argparse, subprocess, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_fits_wrapper import build_generate_command

p = argparse.ArgumentParser()
p.add_argument("--csv", default="/ocean/projects/phy250048p/shared/samples/test_100k_simv3_cosi_xu3_tf.csv")
p.add_argument("--rows", type=int, nargs="+", required=True)
p.add_argument("--tag", default="dev_fiberpilot")
a = p.parse_args()
df = pd.read_csv(a.csv, index_col=None)
for layout in ("galaxy_axis", "measured_axis"):
    t0 = time.time()
    for r in a.rows:
        row = df.iloc[r].copy(); row["fiber_layout"] = layout
        subprocess.run(build_generate_command(row, part=1, dataset=f"{a.tag}_{layout}"), check=True,
                       stdout=subprocess.DEVNULL)
    print(layout, f"{(time.time()-t0)/len(a.rows):.1f} s/galaxy", flush=True)
