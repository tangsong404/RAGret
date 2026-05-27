from __future__ import annotations

from pathlib import Path


def preprocess_xlsx(path: Path) -> str:
    try:
        import openpyxl
    except ImportError as e:
        raise RuntimeError("openpyxl is required for .xlsx preprocessing") from e

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    parts: list[str] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows: list[str] = []
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else str(v) for v in row]
            if any(c.strip() for c in cells):
                rows.append("\t".join(cells))
        if rows:
            parts.append(f"--- sheet {sheet_name} ---\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(parts).strip()
