"""Module containing some utility functions such as for visualisation."""
from typing import Any
import inspect
from IPython.display import HTML, display
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


def display_source(non_builtin_object: Any, style="default"):
    """Display the source code of a function in a Jupyter notebook.

    Useful for pedagogical exposition while not repeating library code.

    Args:
        non_builtin_object: A non-builtin object, e.g. a function.
        style: The style of the source code. See pygments for more information.
    """
    code = inspect.getsource(non_builtin_object)
    html = highlight(code, PythonLexer(), HtmlFormatter(style=style))
    stylesheet = f"<style>{HtmlFormatter(style=style).get_style_defs('.highlight')}</style>"
    display(HTML(f"{stylesheet}{html}"))
