import re
import unicodedata
from pydantic import BaseModel as PydanticBaseModel
from typing import Optional


def to_valid_file_name(value: str) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII. Convert spaces or repeated dashes to single dashes.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]", "_", value).strip("-_")


class CustomNameDisplayer(PydanticBaseModel):
    """
    Parameters:
        display_names: Dictionary for string replacement in figures. All strings not in this dictionary will be displayed as is.
    """

    display_names: Optional[dict] = {}
    """Dictionary for string replacement in figures. Keys are strings to be replaced (like process names, etc.), values are strings to display instead.
    All strings not in this dictionary will be displayed as is.
    """

    def display_name(self, name):
        return self.display_names[name] if name in self.display_names else name
