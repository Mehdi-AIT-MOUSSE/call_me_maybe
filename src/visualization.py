import json
import re
from typing import Any

from rich.align import Align
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

console = Console()


def logo() -> None:
    """
    Renders 'CALL ME MAYBE' in large ASCII text, centered on the screen.
    """
    console.clear()
    # Use a raw string (r"") to prevent W605 invalid escape sequence errors
    ascii_logo = r"""
     ____    _    _       _       __  __ _____
    / ___|  / \  | |     | |     |  \/  | ____|
   | |     / _ \ | |     | |     | |\/| |  _|
   | |___ / ___ \| |___  | |___  | |  | | |___
    \____/_/   \_\_____| |_____| |_|  |_|_____|

        __  __    _  __   __ ____  _____
       |  \/  |  / \ \ \ / /| __ )| ____|
       | |\/| | / _ \ \ V / |  _ \|  _|
       | |  | |/ ___ \ | |  | |_) | |___
       |_|  |_/_/   \_\|_|  |____/|_____|
    """
    centered_logo = Align.center(Text(ascii_logo, style="bold green"))
    console.print(centered_logo)


def visualization(
    data: dict[str, Any], step_time: float, total_time: float
) -> None:
    """
    Renders the data into stacked Rich panels, perfectly centered,
    wrapped in a main outer border with top and bottom margins.
    """
    prompt_val = data.get("prompt", "")
    name_val = data.get("name", "")
    params_val = json.dumps(data.get("parameters", {}))

    data_text = Text()
    data_text.append("prompt: ", style="bold green")
    data_text.append(f"{prompt_val}\n")

    data_text.append("name: ", style="bold green")
    data_text.append(f"{name_val}\n")

    data_text.append("param: ", style="bold green")

    param_str = str(data.get("parameters", {}))
    param_str_highlighted = re.sub(
        r"(\d+\.\d+)", r"[cyan]\1[/cyan]", param_str
    )
    data_text.append(Text.from_markup(param_str_highlighted))

    panel_1 = Panel(
        Align.center(data_text),
        title="[bold white][*** DATA GENERATED ***][/bold white]",
        title_align="center",
        border_style="white"
    )

    json_inner = (
        f'    "prompt": "{prompt_val}", '
        f'"name": "{name_val}",\n'
        f'    "parameters": {params_val}'
    )
    json_str = f"{{\n{json_inner}\n}}"

    panel_2 = Panel(
        Align.center(json_str),
        title="[bold white][*** JSON FORMAT ***][/bold white]",
        title_align="center",
        border_style="white"
    )

    bench_text = Text.from_markup(
        f"[cyan]Step:[/cyan] {step_time:.3f} min   |    "
        f"[cyan]Total:[/cyan] {total_time:.3f} min"
    )
    bench_title = "[green][--- Time ---][/green]"

    panel_3 = Panel(
        Align.center(bench_text),
        title=bench_title,
        title_align="center",
        border_style="green"
    )

    content_group = Group(panel_1, panel_2, panel_3)

    outer_panel = Panel(
        content_group,
        border_style="dim white",
        padding=(1, 2)
    )

    margined_layout = Padding(outer_panel, (2, 0, 2, 0))

    console.print(margined_layout)
