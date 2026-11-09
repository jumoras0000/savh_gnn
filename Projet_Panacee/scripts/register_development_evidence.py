"""Archive localement une pièce du dossier de développement avec SHA-256.

Une pièce enregistrée n'est pas une preuve validée ou une autorisation clinique.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp.translational import EVIDENCE_GATES, register_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate", choices=sorted(EVIDENCE_GATES))
    parser.add_argument("file", type=Path)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    print(json.dumps(register_evidence(ROOT, args.gate, args.file, args.title),
                     ensure_ascii=False))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
