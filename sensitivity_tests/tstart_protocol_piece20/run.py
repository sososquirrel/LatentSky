"""Re-run the t*(lead) calibration on piece 20 (see HYPOTHESIS.md)."""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "experiments" / "tstart_protocol"))

import run_tstart_by_lead as base  # noqa: E402

base.PIECE = 20
base.OUT = HERE / "out"
base.OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    print(f"sensitivity: piece={base.PIECE}, frames={base.FRAMES}", flush=True)
    base.main()
