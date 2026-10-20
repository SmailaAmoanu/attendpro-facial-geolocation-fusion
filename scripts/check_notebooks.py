"""Execute every notebook cell in an in-process IPython shell, without sockets.

This is useful in restricted CI environments where Jupyter kernels cannot open
local sockets. Normal interactive use should still use JupyterLab. --write
refreshes the saved outputs only after every cell in a notebook succeeds.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config
from matplotlib_inline.backend_inline import configure_inline_support

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--notebook", choices=[p.name for p in (ROOT / "notebooks").glob("*.ipynb")]
    )
    args = parser.parse_args()
    if args.notebook is None:
        # A fresh process also resets matplotlib/IPython's display registrations.
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            command = [sys.executable, __file__, "--notebook", path.name]
            if args.write:
                command.append("--write")
            subprocess.run(command, cwd=ROOT, check=True)
        return
    config = Config()
    config.HistoryManager.enabled = False
    for path in [ROOT / "notebooks" / args.notebook]:
        shell = InteractiveShell.instance(config=config)
        shell.reset(new_session=True)
        configure_inline_support(shell, "module://matplotlib_inline.backend_inline")
        notebook = nbformat.read(path, as_version=4)
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(
                    f"{path.name}, code cell {count}: {captured.stdout} {captured.stderr}"
                ) from (result.error_before_exec or result.error_in_exec)
            outputs = []
            if captured.stdout:
                outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stdout", text=captured.stdout
                    )
                )
            if captured.stderr:
                outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stderr", text=captured.stderr
                    )
                )
            for displayed in captured.outputs:
                outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=displayed.data, metadata=displayed.metadata
                    )
                )
            cell.outputs = outputs
            cell.execution_count = count
        nbformat.validate(notebook)
        if args.write:
            nbformat.write(notebook, path)
        plt.close("all")
        print(f"PASSED {path.name}: {count} code cells", flush=True)
        InteractiveShell.clear_instance()


if __name__ == "__main__":
    main()
