from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from pathlib import Path

NOTEBOOKS = [
    "02_bronze_ingestion.ipynb",
    "03_silver.ipynb",
    "05_gold.ipynb",
    "06_gold_ml_features.ipynb",
    "07_model_experiments.ipynb",
    "08_model_registration.ipynb",
    "09_batch_forecasting.ipynb",
    "10_market_segmentation.ipynb",
    "11_growth_classifier.ipynb",
    "12_anomaly_detection.ipynb",
    "13_market_opportunity.ipynb",
]

NOTEBOOK_DIR = Path("notebooks")
MARKER = "# HIGH_GARDEN_CATALOG_PARAMETER"
PARAMETER_CELL = (
    f"{MARKER}\n"
    'dbutils.widgets.text("catalog", "high_garden")\n'
    'catalog = dbutils.widgets.get("catalog").strip() or "high_garden"\n'
    'print(f"Using Unity Catalog: {catalog}")\n'
)


def as_source_list(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)


def parameter_cell() -> dict:
    return {
        "cell_type": "code",
        "execution_count": 0,
        "metadata": {},
        "outputs": [],
        "source": as_source_list(PARAMETER_CELL),
    }


def insertion_index(cells: list[dict]) -> int:
    restart_indexes = [
        index
        for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and source_text(cell).lstrip().startswith("%restart_python")
    ]
    if restart_indexes:
        return max(restart_indexes) + 1

    index = 0
    while index < len(cells):
        cell = cells[index]
        text = source_text(cell).lstrip()
        if cell.get("cell_type") == "code" and text.startswith("%pip"):
            index += 1
            continue
        break
    return index


def rewrite_string_token(token_text: str) -> str:
    try:
        value = ast.literal_eval(token_text)
    except (SyntaxError, ValueError):
        return token_text

    if not isinstance(value, str) or "high_garden." not in value:
        return token_text

    if "{" in value or "}" in value:
        raise ValueError(f"Cannot safely parameterize string containing braces: {value!r}")

    rewritten = value.replace("high_garden.", "{catalog}.")
    return "f" + json.dumps(rewritten)


def rewrite_python(source: str) -> str:
    # Some notebooks define a separate CATALOG constant. Make it reference
    # the job/widget parameter instead of silently falling back to dev.
    source = re.sub(
        r'(?m)^\s*CATALOG\s*=\s*[\"\']high_garden[\"\']\s*$',
        "CATALOG = catalog",
        source,
    )

    stream = io.StringIO(source)
    tokens = []
    for tok in tokenize.generate_tokens(stream.readline):
        if tok.type == tokenize.STRING:
            tok = tokenize.TokenInfo(
                tok.type,
                rewrite_string_token(tok.string),
                tok.start,
                tok.end,
                tok.line,
            )
        tokens.append(tok)
    return tokenize.untokenize(tokens)


def rewrite_sql_magic(source: str) -> str:
    lines = source.splitlines()
    sql = "\n".join(lines[1:]).strip()
    sql = sql.replace("high_garden.", "{catalog}.")
    return f'display(spark.sql(f"""\n{sql}\n"""))\n'


def transform_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])

    if not any(MARKER in source_text(cell) for cell in cells):
        cells.insert(insertion_index(cells), parameter_cell())

    changed = False
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue

        source = source_text(cell)
        stripped = source.lstrip()

        if stripped.startswith("%sql") and "high_garden." in source:
            rewritten = rewrite_sql_magic(source)
        elif stripped.startswith("%"):
            rewritten = source
        else:
            rewritten = rewrite_python(source)

        if rewritten != source:
            cell["source"] = as_source_list(rewritten)
            changed = True

    notebook["cells"] = cells

    remaining = [
        source_text(cell)
        for cell in cells
        if cell.get("cell_type") == "code"
        and (
            "high_garden." in source_text(cell)
            or re.search(
                r'(?m)^\s*CATALOG\s*=\s*[\"\']high_garden[\"\']\s*$',
                source_text(cell),
            )
        )
    ]
    if remaining:
        raise RuntimeError(
            f"Unparameterized high_garden identifiers remain in {path}: {remaining[:3]}"
        )

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return changed


def main() -> None:
    for name in NOTEBOOKS:
        path = NOTEBOOK_DIR / name
        if not path.exists():
            raise FileNotFoundError(path)
        transform_notebook(path)
        print(f"Parameterized {path}")


if __name__ == "__main__":
    main()
