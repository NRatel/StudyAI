#!/usr/bin/env python3
"""Generate standalone HTML previews from the study Markdown files.

The project intentionally avoids a runtime dependency on pandoc or Python
Markdown packages. This converter supports the Markdown subset used by the
study notes: headings, paragraphs, blockquotes, lists, tables, fenced code,
links, inline code, bold text, and LaTeX math rendered by MathJax in browser.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "html"
GENERATED_SOURCES: set[Path] = set()
ROOT_MARKDOWN_FILES = [
    "README.md",
    "AI学习目录索引.md",
    "整体知识体系思维导图.md",
    "附录_数学基础速览.md",
    "附录_代码示例集.md",
]
ROOT_DOC_MODULE = "总览与附录"


STYLE_CSS = r"""
:root {
  color-scheme: light;
  --bg: #f7f8fb;
  --paper: #ffffff;
  --ink: #172033;
  --muted: #667085;
  --line: #d9e0ea;
  --line-soft: #edf1f7;
  --accent: #2563eb;
  --accent-soft: #e8f0ff;
  --accent-2: #0f766e;
  --code-bg: #f3f5f8;
  --code-ink: #1f2937;
  --quote-bg: #f4faf8;
  --shadow: 0 18px 48px rgba(17, 24, 39, 0.08);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 32rem),
    linear-gradient(180deg, #fbfcff 0%, var(--bg) 24rem);
  font-family: "Inter", "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.75;
}

a {
  color: var(--accent);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  min-height: 56px;
  padding: 0.6rem clamp(1rem, 3vw, 2rem);
  border-bottom: 1px solid rgba(217, 224, 234, 0.78);
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(14px);
}

.brand {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  color: var(--ink);
  font-weight: 700;
}

.brand-mark {
  display: inline-grid;
  width: 28px;
  height: 28px;
  place-items: center;
  border-radius: 7px;
  color: #fff;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  font-size: 0.88rem;
}

.crumb {
  min-width: 0;
  overflow: hidden;
  color: var(--muted);
  font-size: 0.92rem;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.shell {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 1.5rem;
  width: min(1220px, calc(100% - 2rem));
  margin: 1.6rem auto 3rem;
}

.doc-shell {
  grid-template-columns: 260px minmax(0, 1fr);
  align-items: start;
}

.toc {
  position: sticky;
  top: 76px;
  max-height: calc(100vh - 96px);
  overflow: auto;
  padding: 1rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.82);
}

.toc-title {
  margin-bottom: 0.55rem;
  color: var(--muted);
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.site-tree details,
.page-toc {
  margin: 0.35rem 0;
}

.site-tree summary,
.page-toc summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.6rem;
  padding: 0.42rem 0.5rem;
  border-radius: 7px;
  color: #1f2937;
  cursor: pointer;
  font-weight: 700;
  line-height: 1.35;
}

.site-tree summary:hover,
.page-toc summary:hover {
  background: #f3f6fb;
}

.site-tree summary::-webkit-details-marker,
.page-toc summary::-webkit-details-marker {
  display: none;
}

.site-tree summary::after,
.page-toc summary::after {
  content: "+";
  flex: 0 0 auto;
  width: 1.2rem;
  height: 1.2rem;
  border: 1px solid var(--line);
  border-radius: 5px;
  color: var(--muted);
  font-size: 0.86rem;
  line-height: 1.1rem;
  text-align: center;
}

.site-tree details[open] > summary::after,
.page-toc[open] > summary::after {
  content: "-";
}

.site-tree .doc-link {
  display: block;
  margin: 0.18rem 0 0.18rem 0.45rem;
  padding: 0.42rem 0.55rem 0.42rem 0.75rem;
  border-left: 2px solid var(--line-soft);
  border-radius: 0 7px 7px 0;
  color: #344054;
}

.site-tree .doc-link small {
  display: block;
  overflow: hidden;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.28;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.site-tree .doc-link.active {
  border-left-color: var(--accent);
  color: var(--accent);
  background: var(--accent-soft);
  font-weight: 700;
}

.sidebar-rule {
  height: 1px;
  margin: 0.9rem 0;
  background: var(--line-soft);
}

.page-toc summary {
  padding-left: 0;
  color: var(--muted);
  font-size: 0.82rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.toc a {
  display: block;
  padding: 0.28rem 0;
  color: #344054;
  font-size: 0.9rem;
  line-height: 1.45;
}

.toc a.depth-3 {
  padding-left: 0.85rem;
  color: var(--muted);
  font-size: 0.84rem;
}

.toc a.depth-4,
.toc a.depth-5,
.toc a.depth-6 {
  padding-left: 1.5rem;
  color: var(--muted);
  font-size: 0.8rem;
}

.page {
  min-width: 0;
  padding: clamp(1.2rem, 4vw, 3.1rem);
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--paper);
  box-shadow: var(--shadow);
}

.markdown-body {
  max-width: 880px;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4 {
  line-height: 1.28;
}

.markdown-body h1 {
  margin: 0 0 1.25rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--line);
  font-size: clamp(2rem, 4vw, 3rem);
  letter-spacing: 0;
}

.markdown-body h2 {
  margin-top: 2.2rem;
  padding-top: 0.35rem;
  font-size: 1.65rem;
}

.markdown-body h3 {
  margin-top: 1.55rem;
  font-size: 1.25rem;
}

.markdown-body h4 {
  margin-top: 1.2rem;
  color: #344054;
  font-size: 1.05rem;
}

.markdown-body p {
  margin: 0.85rem 0;
}

.markdown-body hr {
  height: 1px;
  margin: 2rem 0;
  border: 0;
  background: var(--line);
}

.markdown-body blockquote {
  margin: 1rem 0;
  padding: 0.8rem 1rem;
  border-left: 4px solid var(--accent-2);
  border-radius: 0 8px 8px 0;
  color: #344054;
  background: var(--quote-bg);
}

.markdown-body blockquote p:first-child {
  margin-top: 0;
}

.markdown-body blockquote p:last-child {
  margin-bottom: 0;
}

.markdown-body ul,
.markdown-body ol {
  padding-left: 1.45rem;
}

.markdown-body li {
  margin: 0.28rem 0;
}

.markdown-body code {
  padding: 0.14rem 0.34rem;
  border-radius: 5px;
  color: #b42318;
  background: #fff1f0;
  font-family: "Cascadia Code", Consolas, "SFMono-Regular", monospace;
  font-size: 0.92em;
}

.markdown-body pre {
  overflow: auto;
  margin: 1rem 0;
  padding: 1rem;
  border: 1px solid var(--line);
  border-radius: 10px;
  color: var(--code-ink);
  background: var(--code-bg);
  line-height: 1.55;
}

.markdown-body pre code {
  padding: 0;
  color: inherit;
  background: transparent;
  font-size: 0.9rem;
}

.table-wrap {
  overflow-x: auto;
  margin: 1rem 0;
  border: 1px solid var(--line);
  border-radius: 10px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
}

th,
td {
  padding: 0.62rem 0.78rem;
  border-bottom: 1px solid var(--line-soft);
  text-align: left;
  vertical-align: top;
}

th {
  color: #344054;
  background: #f8fafc;
  font-weight: 700;
}

tr:last-child td {
  border-bottom: 0;
}

.math-block {
  overflow-x: auto;
  margin: 1.05rem 0;
  padding: 0.75rem 0;
}

.doc-nav {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  margin-top: 2.4rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--line);
}

.doc-nav a {
  display: inline-flex;
  max-width: 48%;
  padding: 0.55rem 0.75rem;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
}

.index-hero {
  padding: clamp(1.4rem, 4vw, 2.4rem);
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--paper);
  box-shadow: var(--shadow);
}

.index-hero h1 {
  margin: 0 0 0.6rem;
  font-size: clamp(2rem, 4vw, 3rem);
}

.index-hero p {
  max-width: 760px;
  margin: 0;
  color: var(--muted);
}

.module-card {
  margin-top: 1rem;
  padding: 1.1rem;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--paper);
}

.module-card h2 {
  margin: 0 0 0.75rem;
}

.file-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 0.55rem;
  margin: 0;
  padding: 0;
  list-style: none;
}

.file-list a {
  display: block;
  min-height: 100%;
  padding: 0.68rem 0.78rem;
  border: 1px solid var(--line-soft);
  border-radius: 8px;
  color: var(--ink);
  background: #fbfcff;
}

.file-list a:hover {
  border-color: #bfdbfe;
  background: var(--accent-soft);
  text-decoration: none;
}

@media (max-width: 920px) {
  .doc-shell {
    grid-template-columns: 1fr;
  }

  .toc {
    position: static;
    max-height: none;
  }

  .page {
    padding: 1.2rem;
  }
}

@media print {
  body {
    background: #fff;
  }

  .topbar,
  .toc,
  .doc-nav {
    display: none;
  }

  .shell,
  .doc-shell {
    display: block;
    width: 100%;
    margin: 0;
  }

  .page {
    border: 0;
    box-shadow: none;
  }
}
""".strip()


@dataclass(frozen=True)
class Document:
    source: Path
    output: Path
    title: str
    module: str


@dataclass
class Heading:
    level: int
    text: str
    anchor: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HTML from study Markdown files.")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=["01", "02"],
        help="Module numbers to generate, for example: 01 02. Use --all for every module.",
    )
    parser.add_argument("--all", action="store_true", help="Generate every AI学习_* module.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory first.")
    return parser.parse_args()


def module_dirs(modules: Iterable[str], all_modules: bool) -> list[Path]:
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("AI学习_"))
    if all_modules:
        return dirs

    normalized = {m.zfill(2) for m in modules}
    selected = [p for p in dirs if p.name.split("_", 2)[1] in normalized]
    if not selected:
        raise SystemExit(f"No module directories matched: {', '.join(sorted(normalized))}")
    return selected


def sort_key(path: Path) -> tuple[int, str]:
    if path.name == "README.md":
        return (0, path.name)
    match = re.match(r"^(\d+)_", path.name)
    if match:
        return (int(match.group(1)), path.name)
    if path.name == "论文与FAQ.md":
        return (98, path.name)
    return (99, path.name)


def display_module_name(module: str) -> str:
    match = re.match(r"^AI学习_(\d{2})_(.+)$", module)
    if not match:
        return module
    return f"{match.group(1)} {match.group(2).replace('_', ' ')}"


def display_title(title: str) -> str:
    return re.sub(r"^AI学习[:：]\s*", "", title).strip()


def strip_markdown(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def github_slug(text: str, used: dict[str, int]) -> str:
    base = strip_markdown(text).lower()
    base = re.sub(r"[^\w\s-]", "", base, flags=re.UNICODE)
    base = re.sub(r"\s+", "-", base).strip("-")
    base = base or "section"
    count = used.get(base, 0)
    used[base] = count + 1
    return base if count == 0 else f"{base}-{count}"


def title_from_markdown(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return strip_markdown(line[2:])
    return path.stem


def html_href(url: str, current_source: Path, current_output: Path, output_root: Path) -> str:
    if re.match(r"^(https?:|mailto:|#)", url):
        return url

    target, sep, anchor = url.partition("#")
    if not target:
        return "#" + anchor

    target = target.replace("%20", " ")
    if target.endswith(".md"):
        source_target = (current_source.parent / target).resolve()
        try:
            rel_to_root = source_target.relative_to(ROOT)
            if source_target in GENERATED_SOURCES:
                target_path = (output_root / rel_to_root).with_suffix(".html")
            else:
                target_path = source_target
            rel = os.path.relpath(target_path, current_output.parent)
            href = Path(rel).as_posix()
        except ValueError:
            href = target[:-3] + ".html"
    else:
        href = target

    if sep:
        href += "#" + anchor
    return quote(href, safe="/#:._-%?=&")


def render_inline(text: str, current_source: Path, current_output: Path, output_root: Path) -> str:
    placeholders: list[str] = []

    def stash(value: str) -> str:
        placeholders.append(value)
        return f"\u0000{len(placeholders) - 1}\u0000"

    text = re.sub(r"`([^`]+)`", lambda m: stash(f"<code>{html.escape(m.group(1))}</code>"), text)
    text = html.escape(text)

    def link_repl(match: re.Match[str]) -> str:
        label = match.group(1)
        url = html.unescape(match.group(2).strip())
        href = html.escape(html_href(url, current_source, current_output, output_root), quote=True)
        return f'<a href="{href}">{label}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_repl, text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)

    for i, value in enumerate(placeholders):
        text = text.replace(f"\u0000{i}\u0000", value)
    return text


def is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    first = lines[index].strip()
    second = lines[index + 1].strip()
    if "|" not in first or "|" not in second:
        return False
    cells = [c.strip() for c in second.strip("|").split("|")]
    return bool(cells) and all(re.match(r"^:?-{3,}:?$", c) for c in cells)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render_table(rows: list[str], current_source: Path, current_output: Path, output_root: Path) -> str:
    header = split_table_row(rows[0])
    body = [split_table_row(row) for row in rows[2:]]
    parts = ['<div class="table-wrap"><table><thead><tr>']
    for cell in header:
        parts.append(f"<th>{render_inline(cell, current_source, current_output, output_root)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{render_inline(cell, current_source, current_output, output_root)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def render_sidebar(
    docs: list[Document],
    current_doc: Document | None,
    current_output: Path,
    output_root: Path,
    headings: list[Heading] | None = None,
) -> str:
    modules: dict[str, list[Document]] = {}
    for doc in docs:
        modules.setdefault(doc.module, []).append(doc)

    current_module = current_doc.module if current_doc else None
    module_parts: list[str] = []
    for module, module_docs in modules.items():
        is_open = current_module is None or module == current_module
        module_parts.append(f'<details{" open" if is_open else ""}>')
        module_parts.append(f"<summary>{html.escape(display_module_name(module))}</summary>")
        for doc in module_docs:
            active = current_doc is not None and doc.source == current_doc.source
            href = relative_link(current_output, doc.output)
            module_parts.append(
                f'<a class="doc-link{" active" if active else ""}" href="{href}">'
                f"{html.escape(doc.source.stem)}"
                f"<small>{html.escape(display_title(doc.title))}</small>"
                "</a>"
            )
        module_parts.append("</details>")

    page_toc = ""
    if headings:
        toc_links = []
        for heading in headings:
            if heading.level <= 1:
                continue
            toc_links.append(
                f'<a class="depth-{heading.level}" href="#{html.escape(heading.anchor, quote=True)}">'
                f"{html.escape(heading.text)}</a>"
            )
        if toc_links:
            page_toc = (
                '<div class="sidebar-rule"></div>'
                '<details class="page-toc" open>'
                "<summary>当前页目录</summary>"
                + "\n".join(toc_links)
                + "</details>"
            )

    return (
        '<aside class="toc">'
        '<div class="toc-title">完整目录</div>'
        '<nav class="site-tree">'
        + "\n".join(module_parts)
        + "</nav>"
        + page_toc
        + "</aside>"
    )


def render_list(
    lines: list[str],
    start: int,
    current_source: Path,
    current_output: Path,
    output_root: Path,
) -> tuple[str, int]:
    first = lines[start]
    ordered = bool(re.match(r"^\s*\d+\.\s+", first))
    tag = "ol" if ordered else "ul"
    parts = [f"<{tag}>"]
    i = start
    while i < len(lines):
        line = lines[i]
        pattern = r"^\s*\d+\.\s+(.*)$" if ordered else r"^\s*[-*]\s+(.*)$"
        match = re.match(pattern, line)
        if not match:
            break
        parts.append(f"<li>{render_inline(match.group(1), current_source, current_output, output_root)}</li>")
        i += 1
    parts.append(f"</{tag}>")
    return "\n".join(parts), i


def render_blockquote(
    lines: list[str],
    start: int,
    current_source: Path,
    current_output: Path,
    output_root: Path,
) -> tuple[str, int]:
    parts: list[str] = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith(">"):
        text = re.sub(r"^\s*>\s?", "", lines[i])
        if text:
            parts.append(render_inline(text, current_source, current_output, output_root))
        i += 1
    return "<blockquote><p>" + "<br>\n".join(parts) + "</p></blockquote>", i


def markdown_to_html(source: Path, output: Path, output_root: Path) -> tuple[str, list[Heading]]:
    lines = source.read_text(encoding="utf-8").splitlines()
    parts: list[str] = []
    headings: list[Heading] = []
    used_slugs: dict[str, int] = {}
    paragraph: list[str] = []
    i = 0

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(line.strip() for line in paragraph)
            parts.append(f"<p>{render_inline(text, source, output, output_root)}</p>")
            paragraph.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            language = stripped[3:].strip()
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            class_attr = f' class="language-{html.escape(language)}"' if language else ""
            parts.append(f"<pre><code{class_attr}>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue

        if stripped.startswith("$$"):
            flush_paragraph()
            math_lines = [line]
            i += 1
            if stripped.endswith("$$") and len(stripped) > 2:
                parts.append('<div class="math-block">' + html.escape("\n".join(math_lines)) + "</div>")
                continue
            while i < len(lines):
                math_lines.append(lines[i])
                current = lines[i].strip()
                i += 1
                if current.endswith("$$"):
                    break
            parts.append('<div class="math-block">' + html.escape("\n".join(math_lines)) + "</div>")
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            rendered_text = display_title(text) if level == 1 else text
            anchor = github_slug(rendered_text, used_slugs)
            headings.append(Heading(level, strip_markdown(rendered_text), anchor))
            parts.append(
                f'<h{level} id="{html.escape(anchor, quote=True)}">'
                f"{render_inline(rendered_text, source, output, output_root)}</h{level}>"
            )
            i += 1
            continue

        if re.match(r"^---+$", stripped):
            flush_paragraph()
            parts.append("<hr>")
            i += 1
            continue

        if line.lstrip().startswith(">"):
            flush_paragraph()
            block, i = render_blockquote(lines, i, source, output, output_root)
            parts.append(block)
            continue

        if is_table_start(lines, i):
            flush_paragraph()
            table_lines = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip():
                table_lines.append(lines[i])
                i += 1
            parts.append(render_table(table_lines, source, output, output_root))
            continue

        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            flush_paragraph()
            block, i = render_list(lines, i, source, output, output_root)
            parts.append(block)
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    return "\n".join(parts), headings


def relative_link(from_path: Path, to_path: Path) -> str:
    return quote(os.path.relpath(to_path, from_path.parent).replace(os.sep, "/"), safe="/#:._-%")


def render_doc(doc: Document, docs: list[Document], output_root: Path) -> None:
    body, headings = markdown_to_html(doc.source, doc.output, output_root)
    css_href = relative_link(doc.output, output_root / "assets" / "study.css")
    index_href = relative_link(doc.output, output_root / "index.html")
    current_index = docs.index(doc)
    prev_doc = docs[current_index - 1] if current_index > 0 else None
    next_doc = docs[current_index + 1] if current_index + 1 < len(docs) else None

    sidebar = render_sidebar(docs, doc, doc.output, output_root, headings)

    nav = ['<nav class="doc-nav">']
    if prev_doc:
        nav.append(f'<a href="{relative_link(doc.output, prev_doc.output)}">← {html.escape(display_title(prev_doc.title))}</a>')
    else:
        nav.append("<span></span>")
    if next_doc:
        nav.append(f'<a href="{relative_link(doc.output, next_doc.output)}">{html.escape(display_title(next_doc.title))} →</a>')
    else:
        nav.append("<span></span>")
    nav.append("</nav>")

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(display_title(doc.title))} · AI学习</title>
  <link rel="stylesheet" href="{css_href}">
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <header class="topbar">
    <a class="brand" href="{index_href}"><span class="brand-mark">AI</span><span>AI学习 HTML 预览</span></a>
    <span class="crumb">{html.escape(display_module_name(doc.module))} / {html.escape(doc.source.name)}</span>
  </header>
  <div class="shell doc-shell">
    {sidebar}
    <main class="page">
      <article class="markdown-body">
{body}
      </article>
      {''.join(nav)}
    </main>
  </div>
</body>
</html>
"""
    doc.output.parent.mkdir(parents=True, exist_ok=True)
    doc.output.write_text(html_text, encoding="utf-8", newline="\n")


def render_index(docs: list[Document], output_root: Path) -> None:
    modules: dict[str, list[Document]] = {}
    for doc in docs:
        modules.setdefault(doc.module, []).append(doc)
    module_count = sum(1 for module in modules if module != ROOT_DOC_MODULE)

    cards = []
    for module, module_docs in modules.items():
        items = []
        for doc in module_docs:
            items.append(
                f'<li><a href="{relative_link(output_root / "index.html", doc.output)}">'
                f"{html.escape(doc.source.stem)}<br><small>{html.escape(display_title(doc.title))}</small></a></li>"
            )
        cards.append(
            f'<section class="module-card"><h2>{html.escape(display_module_name(module))}</h2>'
            f'<ul class="file-list">{"".join(items)}</ul></section>'
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI学习 HTML 预览</title>
  <link rel="stylesheet" href="assets/study.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="index.html"><span class="brand-mark">AI</span><span>AI学习 HTML 预览</span></a>
    <span class="crumb">已生成 {module_count} 个模块 + 总览附录，{len(docs)} 个文档</span>
  </header>
  <div class="shell doc-shell">
    {render_sidebar(docs, None, output_root / "index.html", output_root)}
    <main>
      <section class="index-hero">
        <h1>AI学习 HTML 预览</h1>
        <p>这是从 Markdown 生成的 HTML 版本。可通过左侧完整目录浏览总览、附录和各学习模块。</p>
      </section>
      {''.join(cards)}
    </main>
  </div>
</body>
</html>
"""
    (output_root / "index.html").write_text(html_text, encoding="utf-8", newline="\n")


def build_documents(dirs: list[Path], output_root: Path, include_root_docs: bool = False) -> list[Document]:
    docs: list[Document] = []
    if include_root_docs:
        for name in ROOT_MARKDOWN_FILES:
            source = ROOT / name
            if source.exists():
                output = (output_root / source.relative_to(ROOT)).with_suffix(".html")
                docs.append(Document(source=source, output=output, title=title_from_markdown(source), module=ROOT_DOC_MODULE))

    for module_dir in dirs:
        markdown_files = sorted(module_dir.glob("*.md"), key=sort_key)
        for source in markdown_files:
            rel = source.relative_to(ROOT)
            output = (output_root / rel).with_suffix(".html")
            docs.append(Document(source=source, output=output, title=title_from_markdown(source), module=module_dir.name))
    return docs


def main() -> None:
    args = parse_args()
    output_root = args.output if args.output.is_absolute() else ROOT / args.output
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "assets").mkdir(parents=True, exist_ok=True)
    (output_root / "assets" / "study.css").write_text(STYLE_CSS + "\n", encoding="utf-8", newline="\n")

    dirs = module_dirs(args.modules, args.all)
    docs = build_documents(dirs, output_root, include_root_docs=args.all)
    GENERATED_SOURCES.clear()
    GENERATED_SOURCES.update(doc.source.resolve() for doc in docs)
    for doc in docs:
        render_doc(doc, docs, output_root)
    render_index(docs, output_root)

    print(f"Generated {len(docs)} HTML files in {output_root}")
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
