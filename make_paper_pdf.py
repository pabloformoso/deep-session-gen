"""Render PAPER.md → apolloagents_paper.pdf via weasyprint."""

from pathlib import Path
import markdown
import weasyprint

ROOT = Path(__file__).parent
MD_PATH = ROOT / "PAPER.md"
PDF_PATH = ROOT / "apolloagents_paper.pdf"

md_text = MD_PATH.read_text(encoding="utf-8")

# Convert markdown to HTML
body_html = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "codehilite", "toc"],
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --text:      #1a1a2e;
    --subtle:    #4a4a6a;
    --accent:    #3a3aaa;
    --accent2:   #c8a020;
    --border:    #d0d0e0;
    --code-bg:   #f4f4f8;
    --page-bg:   #ffffff;
}

@page {
    size: A4;
    margin: 22mm 20mm 22mm 20mm;
    @bottom-center {
        content: counter(page);
        font-family: 'Inter', sans-serif;
        font-size: 9pt;
        color: var(--subtle);
    }
}

* { box-sizing: border-box; }

body {
    font-family: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    color: var(--text);
    background: var(--page-bg);
    max-width: 100%;
}

/* ── Title block ─────────────────────────────────────────────────────────── */
h1:first-of-type {
    font-size: 22pt;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 6pt 0;
    padding-bottom: 8pt;
    border-bottom: 3px solid var(--accent2);
    line-height: 1.3;
}

/* author / date paragraph immediately after h1 */
h1:first-of-type + p {
    font-size: 9.5pt;
    color: var(--subtle);
    margin: 0 0 20pt 0;
}

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 18pt 0;
}

/* ── Headings ─────────────────────────────────────────────────────────────── */
h2 {
    font-size: 14pt;
    font-weight: 700;
    color: var(--accent);
    margin: 24pt 0 6pt 0;
    padding-bottom: 3pt;
    border-bottom: 1.5px solid var(--border);
    page-break-after: avoid;
}

h3 {
    font-size: 11.5pt;
    font-weight: 600;
    color: var(--text);
    margin: 16pt 0 4pt 0;
    page-break-after: avoid;
}

h4 {
    font-size: 10.5pt;
    font-weight: 600;
    color: var(--subtle);
    margin: 12pt 0 3pt 0;
}

/* ── Body text ───────────────────────────────────────────────────────────── */
p { margin: 0 0 8pt 0; }

strong { font-weight: 600; color: var(--text); }
em     { font-style: italic; }

/* ── Code ────────────────────────────────────────────────────────────────── */
code {
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 8.8pt;
    background: var(--code-bg);
    padding: 1px 4px;
    border-radius: 3px;
    color: #2a2a5a;
}

pre {
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 10pt 12pt;
    margin: 8pt 0 12pt 0;
    font-size: 8pt;
    line-height: 1.5;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background: none;
    padding: 0;
    font-size: inherit;
    color: inherit;
}

/* ── Tables ──────────────────────────────────────────────────────────────── */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 10pt 0 14pt 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}

thead tr {
    background: var(--accent);
    color: #ffffff;
}

thead th {
    padding: 6pt 10pt;
    text-align: left;
    font-weight: 600;
    font-size: 9pt;
}

tbody tr:nth-child(even) { background: #f8f8fc; }
tbody tr:nth-child(odd)  { background: #ffffff; }

tbody td {
    padding: 5pt 10pt;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
}

/* ── Lists ───────────────────────────────────────────────────────────────── */
ul, ol {
    margin: 4pt 0 8pt 0;
    padding-left: 18pt;
}

li { margin-bottom: 3pt; }

/* ── Abstract box ────────────────────────────────────────────────────────── */
h2:first-of-type + p,
h2[id="abstract"] + p {
    background: #f0f0f8;
    border-left: 4px solid var(--accent);
    padding: 10pt 14pt;
    border-radius: 0 4px 4px 0;
    font-size: 10pt;
    color: #2a2a4a;
    margin-bottom: 14pt;
}

/* ── Links ───────────────────────────────────────────────────────────────── */
a { color: var(--accent); text-decoration: none; }

/* ── Page breaks ─────────────────────────────────────────────────────────── */
h2 { page-break-before: auto; }
h2:first-of-type { page-break-before: avoid; }
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ApolloAgents — Technical Paper</title>
<style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>"""

print("Rendering PDF…")
doc = weasyprint.HTML(string=full_html, base_url=str(ROOT))
doc.write_pdf(str(PDF_PATH))
print(f"Saved: {PDF_PATH}")
