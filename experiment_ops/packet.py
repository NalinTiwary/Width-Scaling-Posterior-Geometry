"""Package small results, plots and audit records. Keeps cluster state files local."""
import argparse
import csv
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import StringIO

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip",type=Path);p.add_argument("--out",type=Path,default=Path("results/revision_next"))
    a=p.parse_args(); root=a.out
    if not root.is_dir(): p.error(f"missing {root}")
    wanted=[]
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.is_symlink() or f.suffix not in {".csv",".json",".pdf",".txt"}: continue
        if f.name in {"prior_checkpoints.csv","cold_errors_pointwise.csv","recovered_diagnostics.csv"}: continue
        if any(x.startswith(".") for x in f.relative_to(root).parts): continue
        wanted.append(f)
    if not wanted: p.error("No matching results.")
    manifest=StringIO(); w=csv.writer(manifest); w.writerow(["member","bytes"])
    with ZipFile(a.zip,"x",compression=ZIP_DEFLATED) as z:
        for f in wanted:
            name=f.relative_to(root).as_posix();z.write(f,name);w.writerow([name,f.stat().st_size])
        z.writestr("_packet_manifest.csv",manifest.getvalue())
    print(f"{a.zip}: {len(wanted)} small result files")
if __name__=="__main__": main()
