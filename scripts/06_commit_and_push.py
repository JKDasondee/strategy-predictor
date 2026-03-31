"""Git add, commit, and push everything.
Run: python scripts/06_commit_and_push.py
"""
import subprocess as sp

cmds = [
    ["git", "add", "scripts/", "notebooks/", "docs/", "README.md"],
    ["git", "commit", "-m", "add: scripts, notebooks, README, eval report\n\n"
     "- 6 automation scripts (fetch, train, eval, readme, notebooks, push)\n"
     "- 3 Jupyter notebooks (data EDA, baselines, neural models)\n"
     "- README with architecture diagram and quick start\n"
     "- docs/eval.md model comparison report\n\n"
     "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"],
    ["git", "push"],
]

for cmd in cmds:
    print(f"$ {' '.join(cmd[:3])}...")
    r = sp.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ERROR: {r.stderr.strip()}")
    else:
        print(f"  OK")

print("\nDone. Check https://github.com/JKDasondee/strategy-predictor")
