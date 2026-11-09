"""Helpers shared by the export and plotting classes."""

import re
import unicodedata

from .._base_model import FlodymBaseModel


def to_valid_file_name(value: str) -> str:
    """Convert a string into a valid file name.

    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII. Convert spaces or repeated dashes to single dashes.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace, dashes, and underscores.

    Args:
        value: String to convert.

    Returns:
        The converted string.
    """
    value = str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]", "_", value).strip("-_")


class CustomNameDisplayer(FlodymBaseModel):
    """Base class for plotters that allow replacing names in figures by custom display names."""

    display_names: dict[str, str] = {}
    """Dictionary for string replacement in figures.

    Keys are strings to be replaced (like process names, etc.), values are strings to display
    instead.
    All strings not in this dictionary will be displayed as is.
    """

    def display_name(self, name: str) -> str:
        """Return the display name for ``name``, or ``name`` itself if no replacement is given."""
        return self.display_names.get(name, name)
