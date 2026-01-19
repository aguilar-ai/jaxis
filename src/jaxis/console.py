from __future__ import annotations

from textwrap import dedent
from typing import Any, TypedDict

import rich.console
from rich.align import Align
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TableData(TypedDict):
  title: str
  columns: list[tuple[str, dict[str, Any]]]
  rows: list[tuple[str, ...]]


class PanelData(TypedDict):
  title: str
  content: str
  use_markdown: bool


class Console:
  def __init__(self):
    self.console = rich.console.Console()
    self.panel: PanelData | None = None
    self.tables: list[TableData] = []

  def with_panel(
    self, title: str, content: str, *, use_markdown: bool = True
  ) -> "Console":
    self.panel = {
      "title": title,
      "content": dedent(content),
      "use_markdown": use_markdown,
    }
    return self

  def with_table(
    self,
    title: str,
    columns: list[tuple[str, dict[str, Any]]],
    rows: list[tuple[str, ...]],
  ) -> "Console":
    centered_columns = []
    for name, options in columns:
      if "justify" not in options:
        options = {**options, "justify": "center"}
      centered_columns.append((name, options))
    self.tables.append(
      {
        "title": title,
        "columns": centered_columns,
        "rows": rows,
      }
    )
    return self

  def print(self):
    md = None
    if self.panel is not None:
      md = (
        Markdown(self.panel["content"])
        if self.panel["use_markdown"]
        else self.panel["content"]
      )

    tables: list[Align] = []
    for table in self.tables:
      t = Table(title=table["title"])
      for col in table["columns"]:
        t.add_column(col[0], **col[1])
      for row in table["rows"]:
        t.add_row(*row)
      tables.append(Align.center(t))

    group_items = [md] if md is not None else []
    for table in tables:
      group_items.append(table)
      group_items.append(Text(""))
    if group_items:
      group_items.pop()
    group = Group(*group_items)

    if self.panel is not None:
      self.console.print(Panel(group, title=self.panel["title"]))
      self.console.print("\n")
      self.tables.clear()
      self.panel = None
    else:
      self.console.print(group)
      self.tables.clear()
      self.panel = None
