"""S9 provenance: turn the BioCLIP 1 arXiv HTML (2311.18803) into plain text with
section headings, so that quotes in the S9 report can be checked by grep.

Input : audit/preflight/S9_bioclip1/arxiv_2311.18803v{1,3}.html (downloaded with curl
        from https://arxiv.org/html/2311.18803v3 and .../2311.18803v1 on 2026-10-02).
Output: audit/preflight/S9_bioclip1/arxiv_2311.18803v{1,3}.txt

Each heading line is written as "## <heading text>". Paragraphs, list items, table
cells and figure captions are written one per line. Nothing is summarised.
Run: python scripts/preflight/S9_paper_text.py
"""
import pathlib
import re

from bs4 import BeautifulSoup

ROOT = pathlib.Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight/S9_bioclip1")


def html_to_text(path: pathlib.Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    # Math: keep the alttext of <math> so that symbols do not vanish.
    for m in soup.find_all("math"):
        alt = m.get("alttext", "")
        m.replace_with(f" {alt} ")
    lines = []
    blocks = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "figcaption", "td", "th"])
    for el in blocks:
        # Skip blocks nested in another captured block, to avoid duplicates.
        if el.find_parent(["p", "li", "figcaption", "td", "th"]) is not None and el.name in ("p", "li", "td", "th"):
            continue
        text = " ".join(el.get_text(" ", strip=True).split())
        if not text:
            continue
        if el.name.startswith("h"):
            lines.append("")
            lines.append(f"## {text}")
        else:
            lines.append(text)
    return "\n".join(lines) + "\n"


def main():
    for v in ("v1", "v3"):
        src = ROOT / f"arxiv_2311.18803{v}.html"
        dst = ROOT / f"arxiv_2311.18803{v}.txt"
        txt = html_to_text(src)
        dst.write_text(txt, encoding="utf-8")
        n_lines = txt.count("\n")
        print(f"{src.name} -> {dst.name}: {n_lines} lines")


if __name__ == "__main__":
    main()
