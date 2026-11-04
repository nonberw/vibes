"""Check that every arXiv id cited in docs/LITERATURE.md and paper/refs.bib exists.

Queries the public arXiv API and writes id -> title to docs/refs_verified.txt.
Usage: python scripts/verify_refs.py
"""
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(r"\b(\d{4}\.\d{4,5})\b")


def collect_ids():
    ids = set()
    for p in [ROOT / "docs" / "LITERATURE.md", ROOT / "paper" / "refs.bib"]:
        if p.exists():
            ids |= set(PAT.findall(p.read_text(encoding="utf-8")))
    return sorted(ids)


def query(ids):
    url = "http://export.arxiv.org/api/query?max_results=200&id_list=" + ",".join(ids)
    with urllib.request.urlopen(url, timeout=60) as r:
        root = ET.fromstring(r.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = {}
    for e in root.findall("a:entry", ns):
        eid = e.find("a:id", ns).text.rsplit("/", 1)[-1]
        eid = re.sub(r"v\d+$", "", eid)
        title = " ".join(e.find("a:title", ns).text.split())
        out[eid] = title
    return out


def main():
    ids = collect_ids()
    found = {}
    for i in range(0, len(ids), 40):
        found.update(query(ids[i:i + 40]))
        time.sleep(3)
    lines, missing = [], []
    for i in ids:
        if i in found and found[i].lower() != "error":
            lines.append(f"{i}\t{found[i]}")
        else:
            missing.append(i)
            lines.append(f"{i}\tMISSING")
    (ROOT / "docs" / "refs_verified.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n{len(ids) - len(missing)}/{len(ids)} found")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
