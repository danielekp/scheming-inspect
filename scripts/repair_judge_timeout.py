"""Repair one stopped fixed-followup batch after adding a judge-call deadline.

Dry run is the default. Stop the campaign and its restart wrapper before --apply.
Only judge.py may change. Completed outputs and frozen protocol settings remain
unchanged; the whole incomplete fixed-followup attempt is archived and retried.
"""

import argparse
import json
from pathlib import Path

from scripts import repair_followup_responses as shared

POLICY = shared.RepairPolicy(
    changed_files=frozenset({"scheming_inspect/judge.py"}),
    stage="fixed",
    archive_name="judge-timeout-repair",
    reason=(
        "Enforce the existing judge-call timeout as a hard wall-clock deadline; "
        "retain timed-out judgments as unscored without changing protocol settings"
    ),
)


def repair(directory, *, apply=False, hashes=None, read_log_fn=None):
    return shared.repair(
        directory, apply=apply, hashes=hashes, read_log_fn=read_log_fn, policy=POLICY
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply after stopping all workers")
    args = parser.parse_args(argv)
    print(json.dumps(repair(args.campaign_dir, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
