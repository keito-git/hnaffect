"""Pre-submission checks for both manuscript versions.

1. Every macro used in the sections is defined in numbers.tex (or is a LaTeX command).
2. The jp and en versions use the same set of result macros.
3. Lists every hand-typed number in the text so that each one can be traced to either
   a design constant, a cited source, or a macro (printed for manual review).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "06_paper"
MACRO_DEF = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}")
MACRO_USE = re.compile(r"\\([A-Za-z]+)")
NUMBER = re.compile(r"(?<![\\\w{])\d+(?:[.,]\d+)*\s*(?:\\%|%)?")


def sections(lang: str) -> dict[str, str]:
    return {p.name: p.read_text() for p in sorted((ROOT / lang / "sections").glob("*.tex"))}


def main() -> None:
    defined = set(MACRO_DEF.findall((ROOT / "jp" / "numbers.tex").read_text()))
    used = {}
    for lang in ["jp", "en"]:
        u = set()
        for name, txt in sections(lang).items():
            body = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("%"))
            u |= {m for m in MACRO_USE.findall(body) if m in defined or m[:1].islower() and m[1:2].isupper()}
        used[lang] = u
        undefined = sorted(m for m in u if m not in defined)
        print(f"[{lang}] result macros used: {len(u & defined)}; undefined macro-like names: {undefined}")
    print("macros only in jp:", sorted((used['jp'] - used['en']) & defined))
    print("macros only in en:", sorted((used['en'] - used['jp']) & defined))
    for lang in ["jp", "en"]:
        print(f"\n[{lang}] hand-typed numbers (review each):")
        for name, txt in sections(lang).items():
            for i, line in enumerate(txt.splitlines(), 1):
                if line.lstrip().startswith("%") or "\\cite" in line and not NUMBER.search(line.split("\\cite")[0]):
                    pass
                nums = [n for n in NUMBER.findall(re.sub(r"\\(cite|ref|label|eqref|includegraphics)\{[^}]*\}|\$[^$]*\$", "", line))]
                if nums:
                    print(f"  {name}:{i}: {sorted(set(n.strip() for n in nums))}")


if __name__ == "__main__":
    main()
