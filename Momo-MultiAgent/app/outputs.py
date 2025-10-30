# app/outputs.py
from __future__ import annotations

# Convert Markdown -> HTML
def md_to_html(md: str) -> str:
    try:
        import markdown2  # ensure requirement is installed
    except ImportError as e:
        raise RuntimeError("markdown2 is not installed. Run: pip install markdown2") from e
    return markdown2.markdown(md or "")

# Convert Markdown -> very simple PDF (text only)
def md_to_pdf(md: str, path: str = "out.pdf") -> str:
    try:
        from reportlab.pdfgen import canvas
    except ImportError as e:
        raise RuntimeError("reportlab is not installed. Run: pip install reportlab") from e

    c = canvas.Canvas(path)
    y = 800
    # extremely simple: write plain text lines (no Markdown formatting)
    text = (md or "").splitlines()
    for line in text:
        c.drawString(40, y, line[:110])
        y -= 14
        if y < 50:
            c.showPage()
            y = 800
    c.save()
    return path

__all__ = ["md_to_html", "md_to_pdf"]
