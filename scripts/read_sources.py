from pathlib import Path

from pypdf import PdfReader


ROOT = Path("/Users/huahaowen/Documents/Codex/2026-06-24/6-30-23-59-1-pdf")
OUT = ROOT / "work" / "eda_final" / "source_text"
OUT.mkdir(parents=True, exist_ok=True)


def extract_pdf(path: Path, out_name: str) -> None:
    reader = PdfReader(path)
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        chunks.append(f"\n===== PAGE {page_no} =====\n")
        chunks.append(page.extract_text() or "")
    (OUT / out_name).write_text("".join(chunks), encoding="utf-8")


extract_pdf(Path("/Users/huahaowen/Downloads/探索性数据分析作业8.pdf"), "homework8.txt")
