"""Print information of classes with __str__."""

import math

MAX_STR_LENGTH = 70


def _print_title(title):
    """Print title of the class."""
    str_sides = [
        math.floor(0.5 * (MAX_STR_LENGTH - len(title)) - 1),
        math.ceil(0.5 * (MAX_STR_LENGTH - len(title)) - 1),
    ]
    output_str = "".join(["-"] * MAX_STR_LENGTH)
    output_str += (
        "\n" + "".join(["-"] * str_sides[0]) + " " + title + " " + "".join(["-"] * str_sides[1])
    )
    output_str += "\n" + "".join(["-"] * MAX_STR_LENGTH)
    return output_str


def _print_subtitle(subtitle):
    """Print a subtitle."""
    str_sides = [
        math.floor(0.5 * (MAX_STR_LENGTH - len(subtitle)) - 1),
        math.ceil(0.5 * (MAX_STR_LENGTH - len(subtitle)) - 1),
    ]
    return "".join([" "] * str_sides[0]) + " " + subtitle + " " + "".join([" "] * str_sides[1])


def _print_hline():
    """Print horizontal line."""
    return "".join(["-"] * MAX_STR_LENGTH)


def _print_list(title, list0):
    """Print a list."""
    list0 = [str(value) for value in list0]
    if len(list0) > 0:
        output_str = " " + title + " - " + list0[0] + "\n"
        for item in list0[1:]:
            output_str += "".join([" "] * len(title)) + "  - " + item + "\n"
    else:
        output_str = " " + title + " not set.\n"
    return output_str


def _print_dict(title, dict0, inline=False, float_precision=None):
    """Print a dict."""
    if float_precision is None:
        list0 = [str(key) + ": " + str(value) for key, value in dict0.items()]
    else:
        list0 = []
        for key, value in dict0.items():
            val_str = str(key) + ": "
            if isinstance(value, float):
                value = f"{value:.{float_precision}f}"
                val_str += " ".join([""] * (10 + float_precision - len(value))) + value + ","
            else:
                val_str += value + ","
            list0.append(val_str)
    if len(list0) > 0:
        output_str = " " + title
        if not inline:
            output_str += "\n"
        for item in list0:
            output_str += "   " + item
            if not inline:
                output_str += "\n"
    else:
        output_str = " " + title + " not set.\n"
    return output_str
