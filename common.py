from rich.console import Console
from rich.table import Table


def print_table(data: list[dict]) -> None:
    """Print a list of dictionaries as a formatted table using Rich library.

    :param data: List of dictionaries where each dictionary represents a row with key-value pairs as column-value.
    :return: None
    """
    console = Console()

    if not data:
        return

    # Collect all unique keys from all dictionaries to be columns
    columns = set()
    for d in data:
        columns.update(d.keys())
    columns = sorted(list(columns))

    table = Table(show_header=True, header_style="bold magenta")

    # Add a column for row number at the start
    table.add_column("Doc. No.", style="dim", width=8)

    # Add other columns after
    for key in columns:
        table.add_column(key)

    # Add rows with row number as first column, then values corresponding to keys
    for idx, item in enumerate(data, start=1):
        row = [str(idx)]  # Row number as string
        row.extend(str(item.get(key, "")) for key in columns)
        table.add_row(*row)

    console.print(table)


def filter_positive_scores_by_index(scores: dict[str, float]):
    """Given a dictionary of lists 'scores', return a list of dictionaries,
    where each dictionary corresponds to the values at a particular index
    filtered to include only keys with values greater than zero.
    """
    return [
        {key: value[i] for key, value in scores.items() if value[i] > 0}
        for i in range(len(next(iter(scores.values()))))
    ]
