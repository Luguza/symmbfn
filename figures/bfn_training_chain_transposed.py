"""Write an editable SVG of the unrolled BFN training chain, styled after Graves et al. (2025), Fig. 1.

Time runs downwards: one band per shown step, with the continuous variable in the left panel and the discrete one in
the right. Step i's band reads theta_{i-1} from above. Its top row runs network -> output -> receiver, then the KL
link to Alice's sender, whose column also holds the data. Below theta_{i-1}'s branch to the network, the update turns
theta_{i-1} and Alice's sample y into theta_i, which ends the band. The prior theta_0 stands above the first band.
Distribution icons come from real Bayesian updates; the network output is simulated.
Labels are plain-text placeholders, to be replaced with TexText objects in Inkscape.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

OUTPUT_PATH = Path(__file__).with_suffix(".svg")
SEED = 13

STEP_COUNT = 4
SHOWN_STEPS = (1, 2, STEP_COUNT)

DATA_VALUE = 0.8
SIGMA_ONE = 0.02
CONTINUOUS_AXIS = (-2.5, 2.5)

CLASS_COUNT = 3
TRUE_CLASS = 1
BETA_ONE = 9.0
STANDARDISED_LOGIT_AXIS = (-5.5, 5.5)

# The simulated network sees context from the other variables, worth this many times the variable's own accuracy.
CONTEXT_ACCURACY_RATIO = 2.0

# Okabe-Ito colours (colour-blind safe). Discrete rows: one colour per class. Continuous rows: the true value in the
# true class's colour, Bob's beliefs in neutral grey.
BLUE = "#0072B2"
VERMILION = "#D55E00"
BLUISH_GREEN = "#009E73"
DARK_GREY = "#4D4D4D"
CLASS_COLORS = (BLUE, VERMILION, BLUISH_GREEN)  # per class A, B, C; B is the true class
TRUTH_COLOR = VERMILION
BELIEF_COLOR = DARK_GREY
INK = "#000000"
FRAME = "#8C8C8C"
BACKGROUND = "#FFFFFF"
STEP_OUTLINE = "#505050"
PANEL_FILL = "#F6F6F6"
PANEL_OUTLINE = "#8C8C8C"
BOB_FILL = "#D5E0EB"  # muted steel blue
ALICE_FILL = "#EBDCCD"  # muted sand

# Strokes (mm): hairlines for plot frames and ticks, regular lines for arrows and boxes, heavier outlines for grouping.
HAIRLINE_WIDTH = 0.15
LINE_WIDTH = 0.25
CURVE_WIDTH = 0.3
PANEL_OUTLINE_WIDTH = 0.3
STEP_OUTLINE_WIDTH = 0.35
ARROWHEAD_LENGTH = 1.0
ARROWHEAD_HALF_WIDTH = 0.42
JUNCTION_RADIUS = 0.35
# Corner radii shrink from the outermost box inwards.
PANEL_RADIUS = 2.5
STEP_BAND_RADIUS = 2.0
COLUMN_STRIP_RADIUS = 1.5
PROCESS_BOX_RADIUS = 0.8

# Type: one family in two sizes. Labels are 7 pt; text inside boxes, on arrows and under ticks is 6 pt.
FONT_FAMILY = "Times New Roman, STIXGeneral, serif"  # placeholders; TexText replaces them
FONT_SIZE = 2.47  # 7 pt in mm
SMALL_FONT_SIZE = 2.12  # 6 pt in mm
SUBSCRIPT_FONT_SIZE = 0.7 * FONT_SIZE
# Glyph metrics of Times, as fractions of the font size, used to place text by its visible ink rather than its baseline.
CAP_HEIGHT_RATIO = 0.66
X_HEIGHT_RATIO = 0.45
DESCENT_RATIO = 0.22  # depth of parentheses and subscripts below the baseline
# Spacing: every text keeps TEXT_GAP of clear space to what it labels, and every box keeps PADDING between its edge and
# its content. Neighbouring boxes are PADDING apart.
TEXT_GAP = 1.0
PADDING = 1.2

# Horizontal layout (mm). A panel holds one variable. Its columns, left to right: theta (with the update), network,
# output, receiver, then Alice's column with the data and the sender, and the data bus. The gaps are sized so
# that every distribution label, at full width, sits centred above its icon.
ICON_WIDTH = 8.5
ICON_HEIGHT = 7.0
PROCESS_BOX_WIDTH = 8.5  # the network and update boxes; fits the 6 pt word "Network" with a margin
PROCESS_BOX_HEIGHT = 4.5
# Keeps the theta column, and the update box in it when wider than the icons, PADDING inside the Bob box.
THETA_LEFT_MARGIN = 2 * PADDING + max(PROCESS_BOX_WIDTH - ICON_WIDTH, 0.0) / 2
THETA_TO_NETWORK_GAP = 2.3
NETWORK_TO_OUTPUT_GAP = 2.3
OUTPUT_TO_RECEIVER_GAP = 10.2  # fits the output and receiver labels side by side
KL_GAP = 12.0  # lets the receiver label end inside the Bob box; the Bob/Alice boundary runs through its middle
BUS_OFFSET = 2.0
PANEL_RIGHT_PADDING = 2.0  # keeps the data bus inside the Alice box
PANEL_WIDTH = (
    THETA_LEFT_MARGIN + ICON_WIDTH + THETA_TO_NETWORK_GAP + PROCESS_BOX_WIDTH + NETWORK_TO_OUTPUT_GAP + ICON_WIDTH
    + OUTPUT_TO_RECEIVER_GAP + ICON_WIDTH + KL_GAP + ICON_WIDTH + BUS_OFFSET + PANEL_RIGHT_PADDING
)
STEP_LABEL_MARGIN = 5.0  # chosen so that the margin left of the step labels matches the one right of the bands
PANEL_LEFTS = (STEP_LABEL_MARGIN, STEP_LABEL_MARGIN + PANEL_WIDTH + PADDING)
STEP_BAND_OVERHANG = 1.0  # the step bands reach this far past the outer panels, on both sides
STEP_BAND_LEFT = PANEL_LEFTS[0] - STEP_BAND_OVERHANG
STEP_BAND_RIGHT = PANEL_LEFTS[1] + PANEL_WIDTH + STEP_BAND_OVERHANG
STEP_LABEL_X = STEP_BAND_LEFT - TEXT_GAP  # baseline of the rotated step label, whose letters extend leftwards from it
FIGURE_WIDTH = 160.0

# Vertical layout (mm). Time runs downwards: the prior, then one band per shown step, each with a top row (network to
# sender) and a bottom row (theta_i), with the update between them. The headings stack with PADDING between them.
PANEL_TOP = 0.6
TITLE_Y = PANEL_TOP + PADDING + CAP_HEIGHT_RATIO * FONT_SIZE  # baseline
COLUMN_STRIP_TOP = TITLE_Y + DESCENT_RATIO * FONT_SIZE + PADDING
ROLE_Y = COLUMN_STRIP_TOP + PADDING + CAP_HEIGHT_RATIO * FONT_SIZE  # baseline of the Bob and Alice headings
DATA_ICON_Y = ROLE_Y + DESCENT_RATIO * FONT_SIZE + PADDING
PRIOR_ROW_Y = DATA_ICON_Y  # the prior stands level with the data
TICK_LENGTH = 0.6
TICK_LABEL_DROP = TICK_LENGTH + TEXT_GAP + CAP_HEIGHT_RATIO * SMALL_FONT_SIZE  # baseline of a tick label, below the icon
INPUT_TO_STEP_GAP = TICK_LABEL_DROP + PADDING  # clears the data's class letters, which sit level with the prior
STEP_ROW_GAP = 7.0  # between a band's top and bottom rows; holds the update above theta_i
UPDATE_TO_THETA_GAP = 2.5  # keeps the update, and Alice's arrow into it, low and clear of the top row's ticks
STEP_BAND_TOP_PADDING = PADDING + (CAP_HEIGHT_RATIO + DESCENT_RATIO) * FONT_SIZE + TEXT_GAP  # holds the labels
STEP_BAND_BOTTOM_PADDING = TICK_LENGTH + PADDING
STEP_BAND_HEIGHT = STEP_BAND_TOP_PADDING + 2 * ICON_HEIGHT + STEP_ROW_GAP + STEP_BAND_BOTTOM_PADDING
STEP_BAND_SPACING = PADDING
ELLIPSIS_HEIGHT = 8.0
ELLIPSIS_ARROW_LENGTH = 2.4
ELLIPSIS_DOT_SPACING = 0.9
ELLIPSIS_DOT_RADIUS = 0.25
BOTTOM_MARGIN = PANEL_TOP + 2 * PADDING  # the last band keeps PADDING to the bottom of the Bob and Alice boxes

# Axes: ticks only, and class letters under the data's bars.
VALUE_TICKS = (-2.0, -1.0, 0.0, 1.0, 2.0)
LOGIT_TICKS = (-4.0, -2.0, 0.0, 2.0, 4.0)
CLASS_NAMES = ("A", "B", "C")
LOGIT_SAMPLE_COUNT = 200
STANDARD_NORMAL_PEAK = 1 / math.sqrt(2 * math.pi)
PROBABILITY_FILL_RATIO = 0.88  # a bar of probability 1 fills this much of its frame
# All continuous densities share one linear axis, so equal areas look equal; taller peaks are clipped at the frame.
DENSITY_AT_FRAME_TOP = 5.0
DIRAC_LINE_WIDTH = 0.15
# Logit icons, as in Graves et al. Fig. 1: one slot per class, the density of y_k drawn sideways from the slot's left
# edge, so the vertical axis is the logit and the mean shifts each curve up or down.
LOGIT_BASELINE_INSET = 0.35
LOGIT_WIDTH_RATIO = 0.8  # how much of its slot the peak of a standard normal fills

CHAIN_THETA_INDICES = ("0", "1", "2", "n−1", "n")
# Input and output distributions are over the data x; sender and receiver distributions over the sample y.
DATA_VARIABLE = "x"
SAMPLE_VARIABLE = "y"


@dataclass(frozen=True)
class ContinuousStep:
    previous_mean: float
    previous_precision: float
    prediction: float
    accuracy: float
    mean: float
    precision: float


@dataclass(frozen=True)
class DiscreteStep:
    previous_probabilities: list[float]
    output: list[float]
    accuracy: float
    probabilities: list[float]


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    width: float
    height: float

    @property
    def centre_x(self) -> float:
        return self.x + self.width / 2

    @property
    def centre_y(self) -> float:
        return self.y + self.height / 2

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass(frozen=True)
class Panel:
    """Horizontal positions of one variable's columns."""

    left: float

    @property
    def theta_x(self) -> float:
        return self.left + THETA_LEFT_MARGIN

    @property
    def network_x(self) -> float:
        return self.theta_x + ICON_WIDTH + THETA_TO_NETWORK_GAP

    @property
    def output_x(self) -> float:
        return self.network_x + PROCESS_BOX_WIDTH + NETWORK_TO_OUTPUT_GAP

    @property
    def receiver_x(self) -> float:
        return self.output_x + ICON_WIDTH + OUTPUT_TO_RECEIVER_GAP

    @property
    def alice_x(self) -> float:
        return self.receiver_x + ICON_WIDTH + KL_GAP

    @property
    def bus_x(self) -> float:
        return self.alice_x + ICON_WIDTH + BUS_OFFSET

    @property
    def bob_alice_boundary_x(self) -> float:
        return self.alice_x - KL_GAP / 2

    @property
    def bob_centre_x(self) -> float:
        return (self.theta_x + self.receiver_x + ICON_WIDTH) / 2

    @property
    def centre_x(self) -> float:
        return self.left + PANEL_WIDTH / 2


CONTINUOUS_PANEL = Panel(PANEL_LEFTS[0])
DISCRETE_PANEL = Panel(PANEL_LEFTS[1])


# ---------- BFN math ----------


def softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exponentials = [math.exp(logit - top) for logit in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def sender_logit_means(accuracy: float, class_index: int) -> list[float]:
    return [accuracy * (CLASS_COUNT * (1 if k == class_index else 0) - 1) for k in range(CLASS_COUNT)]


def continuous_accuracy(step: int) -> float:
    return SIGMA_ONE ** (-2 * step / STEP_COUNT) * (1 - SIGMA_ONE ** (2 / STEP_COUNT))


def discrete_accuracy(step: int) -> float:
    return BETA_ONE * (2 * step - 1) / STEP_COUNT**2


def run_continuous_chain(rng: random.Random) -> list[ContinuousStep]:
    mean, precision = 0.0, 1.0
    context_mean, context_precision = 0.0, 0.0
    steps: list[ContinuousStep] = []
    for step in range(1, STEP_COUNT + 1):
        prediction = (precision * mean + context_precision * context_mean) / (precision + context_precision)
        accuracy = continuous_accuracy(step)
        sample = rng.gauss(DATA_VALUE, 1 / math.sqrt(accuracy))
        new_precision = precision + accuracy
        new_mean = (precision * mean + accuracy * sample) / new_precision
        steps.append(ContinuousStep(mean, precision, prediction, accuracy, new_mean, new_precision))
        mean, precision = new_mean, new_precision
        context_accuracy = CONTEXT_ACCURACY_RATIO * accuracy
        context_sample = rng.gauss(DATA_VALUE, 1 / math.sqrt(context_accuracy))
        context_mean = (context_precision * context_mean + context_accuracy * context_sample) / (
            context_precision + context_accuracy
        )
        context_precision += context_accuracy
    return steps


def run_discrete_chain(rng: random.Random) -> list[DiscreteStep]:
    probabilities = [1 / CLASS_COUNT] * CLASS_COUNT
    context_logits = [0.0] * CLASS_COUNT
    steps: list[DiscreteStep] = []
    for step in range(1, STEP_COUNT + 1):
        output = softmax([math.log(p) + c for p, c in zip(probabilities, context_logits)])
        accuracy = discrete_accuracy(step)
        spread = math.sqrt(accuracy * CLASS_COUNT)
        sample = [rng.gauss(mean, spread) for mean in sender_logit_means(accuracy, TRUE_CLASS)]
        new_probabilities = softmax([math.log(p) + y for p, y in zip(probabilities, sample)])
        steps.append(DiscreteStep(probabilities, output, accuracy, new_probabilities))
        probabilities = new_probabilities
        context_accuracy = CONTEXT_ACCURACY_RATIO * accuracy
        context_spread = math.sqrt(context_accuracy * CLASS_COUNT)
        context_means = sender_logit_means(context_accuracy, TRUE_CLASS)
        context_logits = [c + rng.gauss(m, context_spread) for c, m in zip(context_logits, context_means)]
    return steps


def normal_density(value: float, mean: float, deviation: float) -> float:
    return math.exp(-0.5 * ((value - mean) / deviation) ** 2) / (deviation * math.sqrt(2 * math.pi))


# ---------- SVG primitives ----------


class Layers:
    """Collects SVG elements into named Inkscape layers, drawn in the order of NAMES."""

    NAMES = (
        "background",
        "panels",
        "steps",
        "boxes",
        "plot frames",
        "plots",
        "reference lines",
        "arrows",
        "axes",
        "labels",
    )
    HIDDEN = ("reference lines",)

    def __init__(self) -> None:
        self.elements: dict[str, list[str]] = {name: [] for name in self.NAMES}

    def add(self, layer: str, element: str) -> None:
        self.elements[layer].append(element)

    def to_svg(self) -> str:
        groups = "\n".join(self.layer_svg(name) for name in self.NAMES)
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
            f'width="{FIGURE_WIDTH}mm" height="{figure_height()}mm" viewBox="0 0 {FIGURE_WIDTH} {figure_height()}">\n'
            f"{groups}\n</svg>\n"
        )

    def layer_svg(self, name: str) -> str:
        visibility = ' style="display:none"' if name in self.HIDDEN else ""
        return (
            f'<g inkscape:groupmode="layer" inkscape:label="{name}" id="layer-{name.replace(" ", "-")}"{visibility}>\n'
            + "\n".join(self.elements[name])
            + "\n</g>"
        )


def rect(box: Box, stroke: str, stroke_width: float, corner_radius: float = 0.0, fill: str = "none") -> str:
    return (
        f'<rect x="{box.x:.3f}" y="{box.y:.3f}" width="{box.width:.3f}" height="{box.height:.3f}" '
        f'rx="{corner_radius}" style="fill:{fill};stroke:{stroke};stroke-width:{stroke_width}"/>'
    )


def polyline(points: list[tuple[float, float]], stroke: str, stroke_width: float, dashed: bool = False) -> str:
    path = "M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in points)
    dash = ";stroke-dasharray:0.7,0.5" if dashed else ""
    return f'<path d="{path}" style="fill:none;stroke:{stroke};stroke-width:{stroke_width}{dash}"/>'


def arrowhead(tip: tuple[float, float], previous: tuple[float, float]) -> str:
    dx, dy = tip[0] - previous[0], tip[1] - previous[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    base_x, base_y = tip[0] - ux * ARROWHEAD_LENGTH, tip[1] - uy * ARROWHEAD_LENGTH
    left = (base_x - uy * ARROWHEAD_HALF_WIDTH, base_y + ux * ARROWHEAD_HALF_WIDTH)
    right = (base_x + uy * ARROWHEAD_HALF_WIDTH, base_y - ux * ARROWHEAD_HALF_WIDTH)
    corners = " L".join(f"{x:.3f},{y:.3f}" for x, y in (tip, left, right))
    return f'<path d="M{corners} Z" style="fill:{INK};stroke:none"/>'


def add_line(layers: Layers, points: list[tuple[float, float]]) -> None:
    layers.add("arrows", polyline(points, INK, LINE_WIDTH))


def add_arrow(layers: Layers, points: list[tuple[float, float]], dashed: bool = False) -> None:
    """Draws a polyline whose last segment ends in a filled arrowhead touching the final point."""
    tip, previous = points[-1], points[-2]
    dx, dy = tip[0] - previous[0], tip[1] - previous[1]
    length = math.hypot(dx, dy)
    line_end = (tip[0] - dx / length * ARROWHEAD_LENGTH * 0.8, tip[1] - dy / length * ARROWHEAD_LENGTH * 0.8)
    layers.add("arrows", polyline(points[:-1] + [line_end], INK, LINE_WIDTH, dashed=dashed))
    layers.add("arrows", arrowhead(tip, previous))


def add_dot(layers: Layers, x: float, y: float, radius: float) -> None:
    layers.add("arrows", f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" style="fill:{INK};stroke:none"/>')


def text_element(
    x: float, y: float, content: str, anchor: str, size: float, font_family: str, rotate: bool = False, color: str = INK
) -> str:
    rotation = f' transform="rotate(-90 {x:.3f} {y:.3f})"' if rotate else ""
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="{anchor}"{rotation} '
        f'style="font-family:{font_family};font-size:{size}px;fill:{color}">{content}</text>'
    )


def add_label(
    layers: Layers, x: float, y: float, content: str, anchor: str = "middle", size: float = FONT_SIZE, rotate: bool = False
) -> None:
    layers.add("labels", text_element(x, y, content, anchor, size, FONT_FAMILY, rotate))


def centred_baseline(centre_y: float, size: float) -> float:
    """Returns the baseline that centres capital letters of this size on centre_y."""
    return centre_y + CAP_HEIGHT_RATIO * size / 2


def add_label_above(layers: Layers, box: Box, content: str) -> None:
    """Centres a label above the box, TEXT_GAP clear of it."""
    add_label(layers, box.centre_x, box.y - TEXT_GAP - DESCENT_RATIO * FONT_SIZE, content)


def add_label_beside(layers: Layers, box: Box, content: str, is_on_right: bool) -> None:
    """Puts a label level with the box, TEXT_GAP clear of its right or left edge."""
    x, anchor = (box.right + TEXT_GAP, "start") if is_on_right else (box.x - TEXT_GAP, "end")
    add_label(layers, x, centred_baseline(box.centre_y, FONT_SIZE), content, anchor=anchor)


def subscripted(base: str, index: str) -> str:
    """Returns SVG text for `base` with `index` as a subscript, e.g. p_O."""
    return f'{base}<tspan baseline-shift="sub" style="font-size:{SUBSCRIPT_FONT_SIZE:.3f}px">{index}</tspan>'


def conditional_label(index: str, variable: str, condition: str) -> str:
    """Returns SVG text for p_index(variable | condition)."""
    return f"{subscripted('p', index)}({variable} | {condition})"


def add_boxed_label(layers: Layers, box: Box, content: str) -> None:
    layers.add("boxes", rect(box, INK, LINE_WIDTH, corner_radius=PROCESS_BOX_RADIUS, fill=BACKGROUND))
    add_label(layers, box.centre_x, centred_baseline(box.centre_y, SMALL_FONT_SIZE), content, size=SMALL_FONT_SIZE)


def add_axis(layers: Layers, box: Box, ticks: list[tuple[float, str]]) -> None:
    """Draws x-axis ticks; non-empty tick labels go under their tick."""
    for x, label in ticks:
        layers.add("axes", polyline([(x, box.bottom), (x, box.bottom + TICK_LENGTH)], INK, HAIRLINE_WIDTH))
        if label:
            layers.add("axes", text_element(x, box.bottom + TICK_LABEL_DROP, label, "middle", SMALL_FONT_SIZE, FONT_FAMILY))


# ---------- plot icons ----------


def value_to_x(box: Box, value: float, axis: tuple[float, float]) -> float:
    low, high = axis
    clipped = min(max(value, low), high)
    return box.x + (clipped - low) / (high - low) * box.width


def sample_axis(axis: tuple[float, float], count: int) -> list[float]:
    low, high = axis
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def curve(box: Box, density: list[float], color: str) -> str:
    """Draws densities sampled evenly across the box width on the shared density axis, clipped at the box top."""
    count = len(density)
    points = [
        (box.x + i / (count - 1) * box.width, box.bottom - min(d / DENSITY_AT_FRAME_TOP, 1.0) * box.height)
        for i, d in enumerate(density)
    ]
    return polyline(points, color, CURVE_WIDTH)


def gaussian_icon(box: Box, mean: float, deviation: float, color: str) -> str:
    density = [normal_density(v, mean, deviation) for v in sample_axis(CONTINUOUS_AXIS, 400)]
    return curve(box, density, color)


def dirac_icon(box: Box, value: float, color: str) -> str:
    """Draws a Dirac delta at `value` as a thin line up to the box top, for its unbounded height."""
    x = value_to_x(box, value, CONTINUOUS_AXIS)
    return polyline([(x, box.bottom), (x, box.y)], color, DIRAC_LINE_WIDTH)


def reference_line(box: Box) -> str:
    x = value_to_x(box, DATA_VALUE, CONTINUOUS_AXIS)
    return polyline([(x, box.y), (x, box.bottom)], TRUTH_COLOR, HAIRLINE_WIDTH, dashed=True)


def bar_icons(box: Box, probabilities: list[float]) -> list[str]:
    slot = box.width / CLASS_COUNT
    bar_width = slot * 0.6
    elements: list[str] = []
    for k, probability in enumerate(probabilities):
        height = max(probability * box.height * PROBABILITY_FILL_RATIO, 0.12)
        bar = Box(box.x + k * slot + (slot - bar_width) / 2, box.bottom - height, bar_width, height)
        elements.append(rect(bar, "none", 0, fill=class_color(k)))
    return elements


def class_marginal_densities(accuracy: float, class_weights: list[float]) -> list[list[float]]:
    """Density of each standardised logit y_k / sigma along STANDARDISED_LOGIT_AXIS, for a mixture over classes
    whose component j has weight class_weights[j]."""
    scale = math.sqrt(accuracy * CLASS_COUNT)
    values = sample_axis(STANDARDISED_LOGIT_AXIS, LOGIT_SAMPLE_COUNT)
    component_means = [[m / scale for m in sender_logit_means(accuracy, j)] for j in range(CLASS_COUNT)]
    return [
        [sum(class_weights[j] * normal_density(v, component_means[j][k], 1) for j in range(CLASS_COUNT)) for v in values]
        for k in range(CLASS_COUNT)
    ]


def class_color(k: int) -> str:
    return CLASS_COLORS[k]


def logit_to_y(box: Box, value: float) -> float:
    low, high = STANDARDISED_LOGIT_AXIS
    clipped = min(max(value, low), high)
    return box.bottom - (clipped - low) / (high - low) * box.height


def logit_slot_icons(box: Box, accuracy: float, class_weights: list[float]) -> list[str]:
    """Draws each class's standardised-logit density sideways in that class's slot."""
    densities = class_marginal_densities(accuracy, class_weights)
    values = sample_axis(STANDARDISED_LOGIT_AXIS, LOGIT_SAMPLE_COUNT)
    slot = box.width / CLASS_COUNT
    elements: list[str] = []
    for k, density in enumerate(densities):
        baseline_x = box.x + k * slot + LOGIT_BASELINE_INSET
        points = [
            (baseline_x + min(d / STANDARD_NORMAL_PEAK, 1.0) * slot * LOGIT_WIDTH_RATIO, logit_to_y(box, v))
            for d, v in zip(density, values)
        ]
        elements.append(polyline(points, class_color(k), CURVE_WIDTH))
    return elements


def one_hot(index: int) -> list[float]:
    return [1.0 if k == index else 0.0 for k in range(CLASS_COUNT)]


def add_value_axis(layers: Layers, box: Box) -> None:
    add_axis(layers, box, [(value_to_x(box, value, CONTINUOUS_AXIS), "") for value in VALUE_TICKS])


def add_class_axis(layers: Layers, box: Box, is_labelled: bool) -> None:
    slot = box.width / CLASS_COUNT
    ticks = [(box.x + (k + 0.5) * slot, name if is_labelled else "") for k, name in enumerate(CLASS_NAMES)]
    add_axis(layers, box, ticks)


def add_logit_axes(layers: Layers, box: Box) -> None:
    """Classes along the bottom; the standardised logit y / sigma, sigma = sqrt(alpha K), up the left edge."""
    add_class_axis(layers, box, is_labelled=False)
    for value in LOGIT_TICKS:
        y = logit_to_y(box, value)
        layers.add("axes", polyline([(box.x, y), (box.x - TICK_LENGTH, y)], INK, HAIRLINE_WIDTH))


# ---------- layout ----------


def chain_positions(column: int) -> tuple[int, int]:
    """Chain positions 0..4 (theta_0, theta_1, theta_2, theta_{n-1}, theta_n) of theta_{i-1} and theta_i for the
    shown step in this column."""
    return (column, column + 1) if column < 2 else (3, 4)


def band_top(column: int) -> float:
    """Top edge of the band of the shown step in this column."""
    if column == 1:
        return band_top(0) + STEP_BAND_HEIGHT + STEP_BAND_SPACING
    return theta_y(chain_positions(column)[0]) + ICON_HEIGHT + INPUT_TO_STEP_GAP


def top_row_y(column: int) -> float:
    """Top edge of the band's top row: network, output, receiver and sender."""
    return band_top(column) + STEP_BAND_TOP_PADDING


def bottom_row_y(column: int) -> float:
    """Top edge of the band's bottom row: the update and the theta_i it produces."""
    return top_row_y(column) + ICON_HEIGHT + STEP_ROW_GAP


def theta_y(position: int) -> float:
    """Top edge of the theta icon at a chain position. Each step's theta_i ends its band; the prior theta_0 and
    theta_{n-1}, which follows the ellipsis, stand alone above the band they feed."""
    if position == 0:
        return PRIOR_ROW_Y
    if position == 3:
        return theta_y(2) + ICON_HEIGHT + ELLIPSIS_HEIGHT
    producing_column = [chain_positions(column)[1] for column in range(len(SHOWN_STEPS))].index(position)
    return bottom_row_y(producing_column)


def figure_height() -> float:
    return band_top(len(SHOWN_STEPS) - 1) + STEP_BAND_HEIGHT + BOTTOM_MARGIN


def theta_box(position: int, panel: Panel) -> Box:
    return Box(panel.theta_x, theta_y(position), ICON_WIDTH, ICON_HEIGHT)


def network_box(panel: Panel, y: float) -> Box:
    return Box(panel.network_x, y + (ICON_HEIGHT - PROCESS_BOX_HEIGHT) / 2, PROCESS_BOX_WIDTH, PROCESS_BOX_HEIGHT)


def output_box(panel: Panel, y: float) -> Box:
    return Box(panel.output_x, y, ICON_WIDTH, ICON_HEIGHT)


def receiver_box(panel: Panel, y: float) -> Box:
    return Box(panel.receiver_x, y, ICON_WIDTH, ICON_HEIGHT)


def sender_box(panel: Panel, y: float) -> Box:
    return Box(panel.alice_x, y, ICON_WIDTH, ICON_HEIGHT)


def update_box(next_theta: Box) -> Box:
    """The update sits in the theta column, just above the theta_i it produces."""
    top = next_theta.y - UPDATE_TO_THETA_GAP - PROCESS_BOX_HEIGHT
    return Box(next_theta.centre_x - PROCESS_BOX_WIDTH / 2, top, PROCESS_BOX_WIDTH, PROCESS_BOX_HEIGHT)


def data_box(panel: Panel) -> Box:
    return Box(panel.alice_x, DATA_ICON_Y, ICON_WIDTH, ICON_HEIGHT)


def step_band(column: int) -> Box:
    return Box(STEP_BAND_LEFT, band_top(column), STEP_BAND_RIGHT - STEP_BAND_LEFT, STEP_BAND_HEIGHT)


# ---------- drawing ----------


def add_plot_frame(layers: Layers, box: Box) -> None:
    layers.add("plot frames", rect(box, FRAME, HAIRLINE_WIDTH, fill=BACKGROUND))


def draw_panel_background(layers: Layers, panel: Panel) -> None:
    """Draws the variable's outlined box, with Bob's columns and Alice's column as two tinted boxes inside it."""
    panel_box = Box(panel.left, PANEL_TOP, PANEL_WIDTH, figure_height() - 2 * PANEL_TOP)
    layers.add("panels", rect(panel_box, PANEL_OUTLINE, PANEL_OUTLINE_WIDTH, corner_radius=PANEL_RADIUS, fill=PANEL_FILL))
    strip_height = panel_box.bottom - PADDING - COLUMN_STRIP_TOP
    bob_left = panel.left + PADDING
    bob_right = panel.bob_alice_boundary_x - PADDING / 2
    alice_left = panel.bob_alice_boundary_x + PADDING / 2
    alice_right = panel_box.right - PADDING
    bob = Box(bob_left, COLUMN_STRIP_TOP, bob_right - bob_left, strip_height)
    alice = Box(alice_left, COLUMN_STRIP_TOP, alice_right - alice_left, strip_height)
    layers.add("panels", rect(bob, "none", 0, corner_radius=COLUMN_STRIP_RADIUS, fill=BOB_FILL))
    layers.add("panels", rect(alice, "none", 0, corner_radius=COLUMN_STRIP_RADIUS, fill=ALICE_FILL))


def draw_step_bands(layers: Layers) -> None:
    layers.add("background", rect(Box(0, 0, FIGURE_WIDTH, figure_height()), "none", 0, fill=BACKGROUND))
    for column, step in enumerate(SHOWN_STEPS):
        band = step_band(column)
        layers.add("steps", rect(band, STEP_OUTLINE, STEP_OUTLINE_WIDTH, corner_radius=STEP_BAND_RADIUS))
        label = "n" if step == STEP_COUNT else str(step)
        add_label(layers, STEP_LABEL_X, band.centre_y, f"i = {label}", rotate=True)


def draw_header(layers: Layers, panel: Panel, variable_name: str) -> None:
    add_label(layers, panel.centre_x, TITLE_Y, variable_name)
    add_label(layers, panel.bob_centre_x, ROLE_Y, "Bob")
    add_label(layers, panel.alice_x + ICON_WIDTH / 2, ROLE_Y, "Alice")


def draw_chain_thetas(layers: Layers, panel: Panel) -> None:
    """Draws the theta icons, each labelled on its right, where its row is free; the chain arrow enters from above."""
    for position, theta_index in enumerate(CHAIN_THETA_INDICES):
        box = theta_box(position, panel)
        add_plot_frame(layers, box)
        add_label_beside(layers, box, conditional_label("I", DATA_VARIABLE, subscripted("θ", theta_index)), is_on_right=True)
    ellipsis_start, ellipsis_end = theta_box(2, panel).bottom, theta_box(3, panel).y
    x = panel.theta_x + ICON_WIDTH / 2
    add_arrow(layers, [(x, ellipsis_start), (x, ellipsis_start + ELLIPSIS_ARROW_LENGTH)])
    middle = (ellipsis_start + ellipsis_end) / 2
    for offset in (-ELLIPSIS_DOT_SPACING, 0.0, ELLIPSIS_DOT_SPACING):
        add_dot(layers, x, middle + offset, ELLIPSIS_DOT_RADIUS)
    add_arrow(layers, [(x, ellipsis_end - ELLIPSIS_ARROW_LENGTH), (x, ellipsis_end)])


def draw_data_bus(layers: Layers, panel: Panel) -> None:
    box = data_box(panel)
    add_plot_frame(layers, box)
    add_label_beside(layers, box, "x", is_on_right=False)
    last_sender = sender_box(panel, top_row_y(len(SHOWN_STEPS) - 1))
    add_line(layers, [(box.right, box.centre_y), (panel.bus_x, box.centre_y), (panel.bus_x, last_sender.centre_y)])


def draw_step(layers: Layers, column: int, panel: Panel, is_labelled: bool) -> None:
    start, end = chain_positions(column)
    previous_theta, next_theta = theta_box(start, panel), theta_box(end, panel)
    y = top_row_y(column)
    network, output = network_box(panel, y), output_box(panel, y)
    receiver, sender = receiver_box(panel, y), sender_box(panel, y)
    update = update_box(next_theta)
    for box in (output, receiver, sender):
        add_plot_frame(layers, box)
    add_boxed_label(layers, network, "Network")
    add_boxed_label(layers, update, "update")

    # The theta column runs straight down: theta_{i-1}, a branch to the network, the update, theta_i.
    centre_y = network.centre_y
    chain_x = previous_theta.centre_x
    add_arrow(layers, [(chain_x, previous_theta.bottom), (chain_x, centre_y), (network.x, centre_y)])
    add_dot(layers, chain_x, centre_y, JUNCTION_RADIUS)
    add_arrow(layers, [(chain_x, centre_y), (chain_x, update.y)])
    add_arrow(layers, [(chain_x, update.bottom), (chain_x, next_theta.y)])

    add_arrow(layers, [(network.right, centre_y), (output.x, centre_y)])
    add_arrow(layers, [(output.right, centre_y), (receiver.x, centre_y)])
    add_arrow(layers, [(receiver.right, centre_y), (sender.x, centre_y)], dashed=True)
    add_arrow(layers, [(panel.bus_x, centre_y), (sender.right, centre_y)])
    is_last_step = column == len(SHOWN_STEPS) - 1
    if not is_last_step:
        add_dot(layers, panel.bus_x, centre_y, JUNCTION_RADIUS)
    add_arrow(layers, [(sender.centre_x, sender.bottom), (sender.centre_x, update.centre_y), (update.right, update.centre_y)])

    # Step i's conditioning, as in the n-step loss of Graves et al. (2025), Section 3.7.
    previous_index, step_index = CHAIN_THETA_INDICES[start], CHAIN_THETA_INDICES[end]
    previous_theta, previous_time = subscripted("θ", previous_index), subscripted("t", previous_index)
    accuracy = subscripted("α", step_index)
    labels = (
        (output, conditional_label("O", DATA_VARIABLE, f"{previous_theta}; {previous_time}")),
        (receiver, conditional_label("R", SAMPLE_VARIABLE, f"{previous_theta}; {previous_time}, {accuracy}")),
        (sender, conditional_label("S", SAMPLE_VARIABLE, f"{DATA_VARIABLE}; {accuracy}")),
    )
    for box, label in labels:
        add_label_above(layers, box, label)

    add_label(layers, panel.bob_alice_boundary_x, centre_y - TEXT_GAP, "KL", size=SMALL_FONT_SIZE)
    if is_labelled:
        sample_label_y = update.centre_y + TEXT_GAP + X_HEIGHT_RATIO * SMALL_FONT_SIZE
        add_label(layers, sender.centre_x - TEXT_GAP, sample_label_y, "y", anchor="end", size=SMALL_FONT_SIZE)


def draw_continuous_icons(layers: Layers, column: int, step: ContinuousStep) -> None:
    panel = CONTINUOUS_PANEL
    start, end = chain_positions(column)
    y = top_row_y(column)
    deviation = 1 / math.sqrt(step.accuracy)
    previous, current = theta_box(start, panel), theta_box(end, panel)
    sender, output, receiver = sender_box(panel, y), output_box(panel, y), receiver_box(panel, y)
    layers.add("plots", gaussian_icon(sender, DATA_VALUE, deviation, TRUTH_COLOR))
    layers.add("plots", dirac_icon(output, step.prediction, BELIEF_COLOR))
    layers.add("plots", gaussian_icon(receiver, step.prediction, deviation, BELIEF_COLOR))
    if start in (0, 3):
        layers.add("plots", gaussian_icon(previous, step.previous_mean, 1 / math.sqrt(step.previous_precision), BELIEF_COLOR))
    layers.add("plots", gaussian_icon(current, step.mean, 1 / math.sqrt(step.precision), BELIEF_COLOR))
    for box in (output, receiver, previous, current):
        layers.add("reference lines", reference_line(box))
    for box in (sender, output, receiver, previous, current):
        add_value_axis(layers, box)


def draw_discrete_icons(layers: Layers, column: int, step: DiscreteStep) -> None:
    panel = DISCRETE_PANEL
    start, end = chain_positions(column)
    y = top_row_y(column)
    sender, output, receiver = sender_box(panel, y), output_box(panel, y), receiver_box(panel, y)
    elements = logit_slot_icons(sender, step.accuracy, one_hot(TRUE_CLASS))
    elements += bar_icons(output, step.output)
    elements += logit_slot_icons(receiver, step.accuracy, step.output)
    if start in (0, 3):
        elements += bar_icons(theta_box(start, panel), step.previous_probabilities)
    elements += bar_icons(theta_box(end, panel), step.probabilities)
    for element in elements:
        layers.add("plots", element)
    for box in (output, theta_box(start, panel), theta_box(end, panel)):
        add_class_axis(layers, box, is_labelled=False)
    add_logit_axes(layers, sender)
    add_logit_axes(layers, receiver)


def draw_data_icons(layers: Layers) -> None:
    continuous_data, discrete_data = data_box(CONTINUOUS_PANEL), data_box(DISCRETE_PANEL)
    layers.add("plots", dirac_icon(continuous_data, DATA_VALUE, TRUTH_COLOR))
    for element in bar_icons(discrete_data, one_hot(TRUE_CLASS)):
        layers.add("plots", element)
    add_value_axis(layers, continuous_data)
    add_class_axis(layers, discrete_data, is_labelled=True)


def build_figure() -> Layers:
    rng = random.Random(SEED)
    continuous_steps = run_continuous_chain(rng)
    discrete_steps = run_discrete_chain(rng)
    layers = Layers()
    draw_step_bands(layers)
    for panel, name in ((CONTINUOUS_PANEL, "Variable 1 (continuous)"), (DISCRETE_PANEL, "Variable 2 (discrete)")):
        draw_panel_background(layers, panel)
        draw_header(layers, panel, name)
        draw_chain_thetas(layers, panel)
        draw_data_bus(layers, panel)
    draw_data_icons(layers)
    for column, step in enumerate(SHOWN_STEPS):
        for panel in (CONTINUOUS_PANEL, DISCRETE_PANEL):
            draw_step(layers, column, panel, is_labelled=column == 0)
        draw_continuous_icons(layers, column, continuous_steps[step - 1])
        draw_discrete_icons(layers, column, discrete_steps[step - 1])
    return layers


def main() -> None:
    OUTPUT_PATH.write_text(build_figure().to_svg(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
