"""Draw the flodym logo from the letters of a font, with flow arrows added to the "f".

The "f" keeps its shape from the font. Its lower end gets a triangular notch pointing up, and the
end of its hook gets a right-pointing arrowhead. Its crossbar is drawn on top as a separate box,
so that flow, box and remaining text can have different colors.

The script finds the parts of the letter in its outline. This works for fonts whose "f" has a flat
lower end, a vertical cut at the end of the hook, and a crossbar that sticks out to the left of
the stem, such as Plus Jakarta Sans (SIL Open Font License).

The script writes to OUTPUT_DIR the "f" alone, padded to a square (flodym_icon_*), the whole word
(flodym_logo_*), and the word followed by SECOND_TEXT in its own color (flodym_logo_ui_*), each
in the light and the dark theme, as SVG and PNG, with transparent background.

Usage:
    python docs/source/pictures/logo/draw_logo.py
"""

import itertools
import urllib.request
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.textpath import text_to_path

# ---------------------------------------------------------------------------------------------
# Text and font
# ---------------------------------------------------------------------------------------------
TEXT = "flodym"
"""Text of the logo. Its first letter becomes the icon."""
SECOND_TEXT = "designer"
"""Text that follows TEXT in its own color, in the flodym_logo_<SECOND_TEXT>_* files."""
FONT_URL = (
    "https://github.com/google/fonts/raw/main/ofl/plusjakartasans/PlusJakartaSans%5Bwght%5D.ttf"
)
"""Variable font, downloaded on first use."""
FONT_WEIGHT = 600
"""Weight of the text; for Plus Jakarta Sans between 200 and 800."""
F_WEIGHT_FACTOR = 1.25
"""Weight of the first letter relative to FONT_WEIGHT. Values above 1 make the icon bolder."""
FONT_CACHE_DIR = Path.home() / ".cache" / "flodym_logo"
FONT_SIZE = 45.5
"""Font size in drawing units (the SVG uses them as pixels). Sets the size of the drawing."""
EXTRA_GAP = 0.03
"""Additional space between the first letter and the rest of the text, relative to FONT_SIZE.

The arrowhead makes the first letter wider. Increase this value if it comes too close to the
next letter.
"""
SECOND_GAP = 0.175
"""Additional space before SECOND_TEXT, relative to FONT_SIZE. 0.175 is one space character."""
ASCENDERS_TO_ARROW_TIP = True
"""Whether to move the tops of tall letters such as "l" and "d" to the height of the arrow tip.

Only the parts above the middle between x-height and ascender height move; the rest of each
letter keeps its shape.
"""

# ---------------------------------------------------------------------------------------------
# Arrows and box
# ---------------------------------------------------------------------------------------------
ARROW_LENGTH = 0.5
"""Length of the arrowhead at the end of the hook, relative to the stem width."""
NOTCH_LENGTH = ARROW_LENGTH
"""Depth of the notch at the lower end of the stem, relative to the stem width."""
CENTER_BOX = True
"""Whether to center the crossbar horizontally on the stem, keeping its width.

If False, the crossbar stays where the font puts it.
"""
BOX_WIDTH_FACTOR = 1.0
"""Width of the box relative to the width of the font's crossbar. Scaled around the box center."""


# ---------------------------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------------------------
class Theme(NamedTuple):
    """Colors of one logo variant.

    Attributes:
        flow: Color of the stem and the hook, including notch and arrowhead.
        box: Color of the crossbar.
        text: Color of the letters after the first.
        second_text: Color of SECOND_TEXT.
    """

    flow: str
    box: str
    text: str
    second_text: str


THEMES = {
    # for white backgrounds: teal-500 and the GUI's grey
    "light": Theme(flow="#00bba7", box="#505050", text="#505050", second_text="#00bba7"),
    # for dark backgrounds: teal-400 and zinc-300, contrast on zinc-900 9.5:1 and 12.0:1
    "dark": Theme(flow="#00d5be", box="#d4d4d8", text="#d4d4d8", second_text="#00d5be"),
}
"""Color variants. Each is written to its own files, named after the key."""

# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------
MARGIN = 0.05
"""Empty space around the drawing, relative to FONT_SIZE."""
OUTPUT_DIR = Path(__file__).parent
PNG_DPI = 300
"""Resolution of the PNG files. One drawing unit corresponds to one point (1/72 inch)."""

Point = tuple[float, float]
Box = tuple[float, float, float, float]
"""Rectangle as minimum x, minimum y, maximum x and maximum y."""


@cache
def font(weight: float) -> FontProperties:
    """Return the font at the given weight.

    Matplotlib draws variable fonts only in their default weight. The font is therefore converted
    to a font with a fixed weight, which is cached next to the downloaded file.
    """
    source = FONT_CACHE_DIR / "PlusJakartaSans[wght].ttf"
    if not source.exists():
        print(f"downloading {FONT_URL}")
        FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FONT_URL, source)
    instance = FONT_CACHE_DIR / f"{source.stem}-{weight:g}.ttf"
    if not instance.exists():
        instancer.instantiateVariableFont(TTFont(source), {"wght": weight}).save(instance)
    return FontProperties(fname=instance)


def text_path(text: str, weight: float) -> MplPath:
    """Return the outline of a text at FONT_SIZE and the given weight, with the baseline at y = 0.

    Ligatures are switched off, so that letter combinations such as "fl" stay separate letters.
    Matplotlib before 3.11 does not shape text and therefore never forms ligatures.
    """
    try:
        vertices, codes = text_to_path.get_text_path(
            font(weight), text, features=("-liga", "-clig")
        )
    except TypeError:
        vertices, codes = text_to_path.get_text_path(font(weight), text)
    return MplPath(np.asarray(vertices) * FONT_SIZE / text_to_path.FONT_SCALE, codes)


def advance_width(letter: str, weight: float) -> float:
    """Return the horizontal space the font reserves for a letter, at FONT_SIZE."""
    font_data = TTFont(font(weight).get_file())
    glyph = font_data.getBestCmap()[ord(letter)]
    return font_data["hmtx"][glyph][0] / font_data["head"].unitsPerEm * FONT_SIZE


@cache
def letter_outlines() -> tuple[MplPath, MplPath, MplPath]:
    """Return the outlines of the first letter, the rest of TEXT and SECOND_TEXT.

    The baseline is at y = 0. All outlines come from one text layout, so the spacing between
    them is the font's own. The first letter is drawn at FONT_WEIGHT times F_WEIGHT_FACTOR; the
    rest moves by the change of its width.
    """
    first_weight = FONT_WEIGHT * F_WEIGHT_FACTOR
    text = text_path(TEXT + SECOND_TEXT, FONT_WEIGHT)
    first = text_path(TEXT[0], FONT_WEIGHT)
    count = len(first.vertices)
    drawn = first.codes != MplPath.CLOSEPOLY  # the vertex of CLOSEPOLY carries no position
    same_codes = np.array_equal(text.codes[:count], first.codes)
    if not same_codes or not np.allclose(text.vertices[:count][drawn], first.vertices[drawn]):
        raise ValueError(f"Could not separate the letter {TEXT[0]!r} from the rest of the text.")
    second_count = len(text_path(SECOND_TEXT, FONT_WEIGHT).vertices) if SECOND_TEXT else 0
    end = len(text.vertices) - second_count
    widening = advance_width(TEXT[0], first_weight) - advance_width(TEXT[0], FONT_WEIGHT)
    shift = (EXTRA_GAP * FONT_SIZE + widening, 0)
    rest = MplPath(text.vertices[count:end] + shift, text.codes[count:end])
    second_shift = (shift[0] + SECOND_GAP * FONT_SIZE, 0)
    second = MplPath(text.vertices[end:] + second_shift, text.codes[end:])
    return text_path(TEXT[0], first_weight), rest, second


def stem_bottom(letter: MplPath) -> tuple[int, float]:
    """Find the flat lower end of the stem.

    Returns:
        Index of the vertex that ends the bottom edge, and the stem width.
    """
    vertices, codes = letter.vertices, letter.codes
    y_min = vertices[:, 1].min()
    for index in range(1, len(vertices)):
        start, end = vertices[index - 1], vertices[index]
        is_flat = np.isclose(start[1], y_min) and np.isclose(end[1], y_min)
        if codes[index] == MplPath.LINETO and is_flat:
            return index, abs(end[0] - start[0])
    raise ValueError("The first letter has no flat lower end.")


def crossbar(letter: MplPath, stem_left: float) -> Box:
    """Find the crossbar: the part of the letter that sticks out to the left of the stem."""
    vertices = letter.vertices[letter.codes != MplPath.CLOSEPOLY]
    left_of_stem = vertices[vertices[:, 0] < stem_left - 1e-6 * FONT_SIZE]
    if len(left_of_stem) == 0:
        raise ValueError("The first letter has no crossbar to the left of the stem.")
    y_min, y_max = left_of_stem[:, 1].min(), left_of_stem[:, 1].max()
    in_band = vertices[(vertices[:, 1] >= y_min - 1e-6) & (vertices[:, 1] <= y_max + 1e-6)]
    return left_of_stem[:, 0].min(), y_min, in_band[:, 0].max(), y_max


def hook_end(letter: MplPath, crossbar_top: float) -> int:
    """Find the vertical cut at the end of the hook: the rightmost vertical edge above the crossbar.

    Returns:
        Index of the vertex that ends this edge.
    """
    vertices, codes = letter.vertices, letter.codes
    best_index, best_x = -1, -np.inf
    for index in range(1, len(vertices)):
        start, end = vertices[index - 1], vertices[index]
        is_vertical = codes[index] == MplPath.LINETO and np.isclose(start[0], end[0])
        if is_vertical and min(start[1], end[1]) > crossbar_top and end[0] > best_x:
            best_index, best_x = index, end[0]
    if best_index < 0:
        raise ValueError("The first letter has no vertical cut at the end of its hook.")
    return best_index


def icon_parts() -> tuple[MplPath, Box]:
    """Return the first letter with notch and arrowhead but without crossbar, and the crossbar.

    The crossbar is cut from the letter by moving its corners onto the edges of the stem, so
    that only the separately drawn box shows it.
    """
    letter, _, _ = letter_outlines()
    bottom_index, stem_width = stem_bottom(letter)
    bottom_start, bottom_end = letter.vertices[bottom_index - 1], letter.vertices[bottom_index]
    stem_left = min(bottom_start[0], bottom_end[0])
    stem_right = max(bottom_start[0], bottom_end[0])
    box = crossbar(letter, stem_left)
    end_index = hook_end(letter, box[3])
    end_start, end_end = letter.vertices[end_index - 1], letter.vertices[end_index]

    notch_tip = ((bottom_start[0] + bottom_end[0]) / 2, bottom_start[1] + NOTCH_LENGTH * stem_width)
    arrow_tip = (end_start[0] + ARROW_LENGTH * stem_width, (end_start[1] + end_end[1]) / 2)
    insertions = {bottom_index: notch_tip, end_index: arrow_tip}

    tolerance = 1e-6 * FONT_SIZE
    vertices: list[Point] = []
    codes: list[int] = []
    for index, (vertex, code) in enumerate(zip(letter.vertices, letter.codes)):
        if index in insertions:
            vertices.append(insertions[index])
            codes.append(MplPath.LINETO)
        x, y = vertex
        if box[1] - tolerance <= y <= box[3] + tolerance:
            x = min(max(x, stem_left), stem_right)
        vertices.append((x, y))
        codes.append(code)

    box_center = (stem_left + stem_right) / 2 if CENTER_BOX else (box[0] + box[2]) / 2
    half_width = BOX_WIDTH_FACTOR * (box[2] - box[0]) / 2
    box = (box_center - half_width, box[1], box_center + half_width, box[3])
    return MplPath(vertices, codes), box


def remaining_texts() -> tuple[MplPath, MplPath]:
    """Return the rest of TEXT and SECOND_TEXT, with ascenders moved to the arrow tip if requested.

    Ascenders are the parts of letters such as "l" and "d" above the middle between x-height and
    the top of the tallest letter of TEXT.
    """
    letter, rest, second = letter_outlines()
    if not ASCENDERS_TO_ARROW_TIP:
        return rest, second
    bottom_index, _ = stem_bottom(letter)
    stem_left = letter.vertices[bottom_index - 1 : bottom_index + 1, 0].min()
    end_index = hook_end(letter, crossbar(letter, stem_left)[3])
    arrow_tip_y = letter.vertices[end_index - 1 : end_index + 1, 1].mean()

    x_height = text_path("x", FONT_WEIGHT).get_extents().y1
    ascender_top = rest.vertices[:, 1].max()
    threshold = (x_height + ascender_top) / 2
    if arrow_tip_y <= threshold:
        raise ValueError("The arrow tip is too low to move the ascenders to it.")
    lowered = []
    for path in (rest, second):
        vertices = path.vertices.copy()
        move = vertices[:, 1] > threshold
        starts = [*np.flatnonzero(path.codes == MplPath.MOVETO), len(vertices)]
        for start, end in itertools.pairwise(starts):
            part = vertices[start:end][path.codes[start:end] != MplPath.CLOSEPOLY, 1]
            if part.min() > x_height:
                # parts entirely above the x-height, such as the dot of an "i", stay in place
                move[start:end] = False
        vertices[move, 1] -= ascender_top - arrow_tip_y
        lowered.append(MplPath(vertices, path.codes))
    return lowered[0], lowered[1]


def text_layers(theme: Theme, with_text: bool, with_second: bool) -> list[tuple[MplPath, str]]:
    """Return the outlines of the letters after the first that are drawn, with their colors."""
    rest, second = remaining_texts()
    layers = [(rest, theme.text)] if with_text else []
    if with_second:
        layers.append((second, theme.second_text))
    return layers


def bounding_box(with_text: bool, with_second: bool) -> Box:
    """Return the extent of the drawing including the margin; a centered square for the icon.

    Args:
        with_text: Whether to include the letters after the first.
        with_second: Whether to include SECOND_TEXT.
    """
    flow, box = icon_parts()
    extents = [flow.get_extents()]
    rest, second = remaining_texts()
    if with_text:
        extents.append(rest.get_extents())
    if with_second:
        extents.append(second.get_extents())
    pad = MARGIN * FONT_SIZE
    x_min = min([box[0]] + [extent.x0 for extent in extents]) - pad
    y_min = min([box[1]] + [extent.y0 for extent in extents]) - pad
    x_max = max([box[2]] + [extent.x1 for extent in extents]) + pad
    y_max = max([box[3]] + [extent.y1 for extent in extents]) + pad
    if not with_text:
        half_size = max(x_max - x_min, y_max - y_min) / 2
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        x_min, x_max = x_center - half_size, x_center + half_size
        y_min, y_max = y_center - half_size, y_center + half_size
    return x_min, y_min, x_max, y_max


def svg_path_data(path: MplPath, to_svg: Callable[[Point], str]) -> str:
    """Convert a matplotlib path to the value of the ``d`` attribute of an SVG path.

    Args:
        path: Path with straight lines and quadratic or cubic Bezier curves.
        to_svg: Converts a point to SVG coordinates.

    Returns:
        The path commands.
    """
    svg_commands = {
        MplPath.MOVETO: "M",
        MplPath.LINETO: "L",
        MplPath.CURVE3: "Q",
        MplPath.CURVE4: "C",
    }
    commands = []
    for vertices, code in path.iter_segments(simplify=False, curves=True):
        if code == MplPath.CLOSEPOLY:
            commands.append("Z")
            continue
        points = [to_svg((x, y)) for x, y in zip(vertices[::2], vertices[1::2])]
        commands.append(f"{svg_commands[code]} {' '.join(points)}")
    return " ".join(commands)


def write_svg(path: Path, theme: Theme, with_text: bool, with_second: bool) -> None:
    """Write the drawing as an SVG file, with exact curves and letter outlines.

    Args:
        path: Target file.
        theme: Colors.
        with_text: Whether to draw the letters after the first.
        with_second: Whether to draw SECOND_TEXT.
    """
    x_min, y_min, x_max, y_max = bounding_box(with_text, with_second)
    width, height = x_max - x_min, y_max - y_min

    def to_svg(point: Point) -> str:
        # SVG has its y axis pointing down
        return f"{point[0] - x_min:.3f},{y_max - point[1]:.3f}"

    flow, box = icon_parts()
    box_element = (
        f'<rect x="{box[0] - x_min:.3f}" y="{y_max - box[3]:.3f}" '
        f'width="{box[2] - box[0]:.3f}" height="{box[3] - box[1]:.3f}" fill="{theme.box}"/>'
    )
    elements = [f'<path d="{svg_path_data(flow, to_svg)}" fill="{theme.flow}"/>', box_element]
    for text, color in text_layers(theme, with_text, with_second):
        elements.append(f'<path d="{svg_path_data(text, to_svg)}" fill="{color}"/>')
    body = "\n  ".join(elements)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.3f}" height="{height:.3f}" '
        f'viewBox="0 0 {width:.3f} {height:.3f}">\n  {body}\n</svg>\n'
    )
    path.write_text(svg, encoding="utf-8")


def write_png(path: Path, theme: Theme, with_text: bool, with_second: bool) -> None:
    """Render the drawing with matplotlib as a PNG file.

    Args:
        path: Target file.
        theme: Colors.
        with_text: Whether to draw the letters after the first.
        with_second: Whether to draw SECOND_TEXT.
    """
    x_min, y_min, x_max, y_max = bounding_box(with_text, with_second)
    # one drawing unit is one point (1/72 inch), as in the SVG
    figure = plt.figure(figsize=((x_max - x_min) / 72, (y_max - y_min) / 72))
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_aspect("equal")
    axes.axis("off")
    flow, box = icon_parts()
    axes.add_patch(PathPatch(flow, facecolor=theme.flow, edgecolor="none"))
    box_width, box_height = box[2] - box[0], box[3] - box[1]
    axes.add_patch(Rectangle(box[:2], box_width, box_height, facecolor=theme.box, edgecolor="none"))
    for text, color in text_layers(theme, with_text, with_second):
        axes.add_patch(PathPatch(text, facecolor=color, edgecolor="none"))
    figure.savefig(path, dpi=PNG_DPI, transparent=True)
    plt.close(figure)


def main() -> None:
    """Write the icon, the logo and the logo with SECOND_TEXT in all themes as SVG and PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = [("icon", False, False), ("logo", True, False)]
    if SECOND_TEXT:
        variants.append((f"logo_{SECOND_TEXT}", True, True))
    for name, with_text, with_second in variants:
        for theme_name, theme in THEMES.items():
            for write, suffix in ((write_svg, "svg"), (write_png, "png")):
                path = OUTPUT_DIR / f"flodym_{name}_{theme_name}.{suffix}"
                write(path, theme, with_text, with_second)
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
