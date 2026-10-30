# Full paper

[Read the full version (PDF)](main.pdf).

This directory contains the complete manuscript, appendix, references, and figures,
preserved from the paper-only repository at commit
`6bf1f5b60e72dafe1833d89889e822bff581e7b4`. `main.pdf` is rebuilt from these sources.

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For Overleaf, upload this directory's contents, choose `main.tex`, and use pdfLaTeX.
