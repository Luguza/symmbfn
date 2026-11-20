"""Write an editable SVG of the unrolled BFN training chain, styled after Graves et al. (2025), Fig. 1.

Each variable (continuous on top, discrete below) has the same lanes. Alice's lane holds the data and the sender of
each step. Bob's lane reads left to right: input theta_{i-1} -> network -> output -> receiver, and above it an update
row that turns theta_{i-1} and Alice's sample y into theta_i. Related distributions share a column: sender above
receiver, update above theta_i. Distribution icons come from real Bayesian updates; the network output is simulated.
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
STEP_FILL = "#F3F3F3"
STEP_OUTLINE = "#505050"
LINE_WIDTH = 0.25
FONT_SIZE = 2.47  # 7 pt in mm
SMALL_FONT_SIZE = 1.76  # 5 pt in mm
ARROWHEAD_LENGTH = 1.0
ARROWHEAD_HALF_WIDTH = 0.42
JUNCTION_RADIUS = 0.35

# Horizontal layout (mm). Within a step, the columns after theta_{i-1} are: network, output, receiver, theta_i.
ICON_WIDTH = 8.0
ICON_HEIGHT = 6.0
GAP = 2.0
COLUMN_PITCH = ICON_WIDTH + GAP
THETA_TO_NETWORK_GAP = 4.4  # wider than GAP: holds two step box borders and the branch towards the update
STEP_BOX_LEFT_OFFSET = 2.0  # step box border, measured right of theta_{i-1}
JUNCTION_OFFSET = 3.1  # branch point of theta_{i-1}'s arrow, measured right of theta_{i-1}
STEP_BLOCK_WIDTH = ICON_WIDTH + THETA_TO_NETWORK_GAP + 3 * COLUMN_PITCH
NETWORK_HEIGHT = 5.0
UPDATE_HEIGHT = 4.0
FIRST_THETA_X = 6.0
ELLIPSIS_WIDTH = 5.0
ELLIPSIS_ARROW_LENGTH = 1.8
LANE_LEFT_X = 4.0
STEP_BOX_OVERHANG = 1.2  # how far a step box reaches past its theta_i, so the p_I(theta_i) label fits inside
LANE_RIGHT_MARGIN = 2.2  # lane borders end 1 mm outside the last step box, not on top of its border
VARIABLE_LABEL_OFFSET = 2.0  # rotated variable name, measured right of the lane borders
FIGURE_WIDTH = 160.0
FIGURE_HEIGHT = 88.0

# Vertical layout (mm). Offsets are relative to the top of a variable's group.
STEP_BOX_TOP_Y = 0.8
HEADER_Y = 4.0
GROUP_TOPS = (7.0, 47.5)
BUS_OFFSET = 1.5
ALICE_ROW_OFFSET = 4.5
ALICE_LANE_HEIGHT = 16.0  # leaves the axis titles of the Alice row clear of the lane border
LANE_SPACING = 1.2
UPDATE_ROW_CENTRE_OFFSET = 20.5
MAIN_ROW_OFFSET = 27.0
GROUP_HEIGHT = 38.5
LABEL_GAP = 1.0
LABEL_SIDE_OFFSET = 0.6  # labels next to a vertical arrow start this far right of it
KL_ARROW_POSITION = 0.64  # across the icon width: right of the axis title and the letter B, left of the tick at 1
SIDE_LABEL_CLEARANCE = 2.8  # a label left of an icon ends this far left of it, clear of the y-axis numbers
KL_LABEL_OFFSET = 19.5
SAMPLE_LABEL_OFFSET = 14.0

# Axes (drawn on the icons of the first step and on the data). Every icon row reserves room for them below.
TICK_LENGTH = 0.6
TICK_LABEL_DROP = 2.4
AXIS_TITLE_DROP = 4.3
LABEL_FONT_FAMILY = "Times New Roman, STIXGeneral, serif"  # placeholders; TexText replaces them
TICK_FONT_FAMILY = "Latin Modern Roman, CMU Serif, Times New Roman, STIXGeneral, serif"
# Numbers go only on the outer ticks, off the centre where the KL arrow enters the sender from below.
VALUE_TICKS = (-2.0, -1.0, 0.0, 1.0, 2.0)
VALUE_LABELLED_TICKS = (-2.0, 2.0)
LOGIT_TICKS = (-4.0, -2.0, 0.0, 2.0, 4.0)
LOGIT_LABELLED_TICKS = (-4.0, 4.0)
CLASS_NAMES = ("A", "B", "C")
LOGIT_SAMPLE_COUNT = 200
STANDARD_NORMAL_PEAK = 1 / math.sqrt(2 * math.pi)
CURVE_FILL_RATIO = 0.88
# Logit icons, as in Graves et al. Fig. 1: one slot per class, the density of y_k drawn sideways from the slot's left
# edge, so the vertical axis is the logit and the mean shifts each curve up or down.
LOGIT_BASELINE_INSET = 0.35
LOGIT_WIDTH_RATIO = 0.8  # how much of its slot the peak of a standard normal fills
Y_AXIS_TITLE_RISE = 0.6  # keeps the y-axis title clear of the data line above the sender

INPUT_NAMES = ("p_I(θ₀)", "p_I(θ₁)", "p_I(θ₂)", "p_I(θₙ₋₁)", "p_I(θₙ)")
UPDATE_ARROW_INSET = 1.0  # the update arrow enters theta this far from its left edge, leaving room for its label
POSITIONS_FED_BY_UPDATE = (1, 2, 4)


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
class Group:
    """Vertical positions of one variable's lanes."""

    top: float

    @property
    def bus_y(self) -> float:
        return self.top + BUS_OFFSET

    @property
    def alice_y(self) -> float:
        return self.top + ALICE_ROW_OFFSET

    @property
    def update_centre_y(self) -> float:
        return self.top + UPDATE_ROW_CENTRE_OFFSET

    @property
    def main_y(self) -> float:
        return self.top + MAIN_ROW_OFFSET

    @property
    def main_centre_y(self) -> float:
        return self.main_y + ICON_HEIGHT / 2


CONTINUOUS_GROUP = Group(GROUP_TOPS[0])
DISCRETE_GROUP = Group(GROUP_TOPS[1])


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
        "steps",
        "lanes and boxes",
        "plot frames",
        "plots",
        "reference lines",
        "network link",
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
            f'width="{FIGURE_WIDTH}mm" height="{FIGURE_HEIGHT}mm" viewBox="0 0 {FIGURE_WIDTH} {FIGURE_HEIGHT}">\n'
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


def add_junction(layers: Layers, x: float, y: float) -> None:
    layers.add("arrows", f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{JUNCTION_RADIUS}" style="fill:{INK};stroke:none"/>')


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
    layers.add("labels", text_element(x, y, content, anchor, size, LABEL_FONT_FAMILY, rotate))


def add_boxed_label(layers: Layers, box: Box, content: str) -> None:
    layers.add("lanes and boxes", rect(box, INK, LINE_WIDTH, corner_radius=0.8, fill=BACKGROUND))
    add_label(layers, box.centre_x, box.centre_y + 0.6, content, size=SMALL_FONT_SIZE)


def add_axis(layers: Layers, box: Box, ticks: list[tuple[float, str]], title: str | None) -> None:
    """Draws x-axis ticks; non-empty tick labels go under their tick and a title, if given, left-aligned below them."""
    for x, label in ticks:
        layers.add("axes", polyline([(x, box.bottom), (x, box.bottom + TICK_LENGTH)], INK, 0.15))
        if label:
            layers.add("axes", text_element(x, box.bottom + TICK_LABEL_DROP, label, "middle", SMALL_FONT_SIZE, TICK_FONT_FAMILY))
    if title is not None:
        layers.add("axes", text_element(box.x, box.bottom + AXIS_TITLE_DROP, title, "start", SMALL_FONT_SIZE, TICK_FONT_FAMILY))


def tick_text(value: float) -> str:
    return f"{value:g}".replace("-", "−")


# ---------- plot icons ----------


def value_to_x(box: Box, value: float, axis: tuple[float, float]) -> float:
    low, high = axis
    clipped = min(max(value, low), high)
    return box.x + (clipped - low) / (high - low) * box.width


def sample_axis(axis: tuple[float, float], count: int) -> list[float]:
    low, high = axis
    return [low + (high - low) * i / (count - 1) for i in range(count)]


def curve(box: Box, density: list[float], peak: float, color: str) -> str:
    """Draws densities sampled evenly across the box width, scaled so that `peak` nearly fills the box height."""
    count = len(density)
    points = [
        (box.x + i / (count - 1) * box.width, box.bottom - min(d / peak, 1.0) * box.height * CURVE_FILL_RATIO)
        for i, d in enumerate(density)
    ]
    return polyline(points, color, 0.3)


def gaussian_icon(box: Box, mean: float, deviation: float, color: str) -> str:
    density = [normal_density(v, mean, deviation) for v in sample_axis(CONTINUOUS_AXIS, 400)]
    return curve(box, density, normal_density(mean, mean, deviation), color)


def delta_icon(box: Box, value: float, color: str) -> str:
    x = value_to_x(box, value, CONTINUOUS_AXIS)
    return polyline([(x, box.bottom), (x, box.bottom - box.height * 0.88)], color, 0.45)


def reference_line(box: Box) -> str:
    x = value_to_x(box, DATA_VALUE, CONTINUOUS_AXIS)
    return polyline([(x, box.y), (x, box.bottom)], TRUTH_COLOR, 0.15, dashed=True)


def bar_icons(box: Box, probabilities: list[float]) -> list[str]:
    slot = box.width / CLASS_COUNT
    bar_width = slot * 0.6
    elements: list[str] = []
    for k, probability in enumerate(probabilities):
        height = max(probability * box.height * 0.88, 0.12)
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
        elements.append(polyline(points, class_color(k), 0.3))
    return elements


def one_hot(index: int) -> list[float]:
    return [1.0 if k == index else 0.0 for k in range(CLASS_COUNT)]


def tick_label(value: float, labelled_values: tuple[float, ...], is_labelled: bool) -> str:
    return tick_text(value) if is_labelled and value in labelled_values else ""


def add_value_axis(layers: Layers, box: Box, is_labelled: bool) -> None:
    ticks = [(value_to_x(box, value, CONTINUOUS_AXIS), tick_label(value, VALUE_LABELLED_TICKS, is_labelled)) for value in VALUE_TICKS]
    add_axis(layers, box, ticks, "value" if is_labelled else None)


def add_class_axis(layers: Layers, box: Box, is_labelled: bool) -> None:
    slot = box.width / CLASS_COUNT
    ticks = [(box.x + (k + 0.5) * slot, name if is_labelled else "") for k, name in enumerate(CLASS_NAMES)]
    add_axis(layers, box, ticks, "class" if is_labelled else None)


def add_logit_axes(layers: Layers, box: Box, is_labelled: bool, has_y_numbers: bool) -> None:
    """Classes along the bottom; the standardised logit y / sigma, sigma = sqrt(alpha K), up the left edge.
    has_y_numbers adds the y tick numbers and title, which need free space left of the icon."""
    add_class_axis(layers, box, is_labelled)
    is_labelled_y = is_labelled and has_y_numbers
    for value in LOGIT_TICKS:
        y = logit_to_y(box, value)
        layers.add("axes", polyline([(box.x, y), (box.x - TICK_LENGTH, y)], INK, 0.15))
        label = tick_label(value, LOGIT_LABELLED_TICKS, is_labelled_y)
        if label:
            layers.add("axes", text_element(box.x - TICK_LENGTH - 0.3, y + 0.6, label, "end", SMALL_FONT_SIZE, TICK_FONT_FAMILY))
    if is_labelled_y:
        layers.add("axes", text_element(box.x, box.y - Y_AXIS_TITLE_RISE, "y/σ", "end", SMALL_FONT_SIZE, TICK_FONT_FAMILY))


# ---------- layout ----------


def theta_x(position: int) -> float:
    """Left edge of the theta icon at chain position 0..4, i.e. theta_0, theta_1, theta_2, theta_{n-1}, theta_n."""
    is_after_ellipsis = position >= 3
    blocks_before = min(position, 2) + max(position - 3, 0)
    x = FIRST_THETA_X + blocks_before * STEP_BLOCK_WIDTH
    return x + (ICON_WIDTH + ELLIPSIS_WIDTH if is_after_ellipsis else 0.0)


def chain_positions(column: int) -> tuple[int, int]:
    """Chain positions of theta_{i-1} and theta_i for the shown step in this column."""
    return (column, column + 1) if column < 2 else (3, 4)


def theta_box(position: int, group: Group) -> Box:
    return Box(theta_x(position), group.main_y, ICON_WIDTH, ICON_HEIGHT)


def network_x(block_x: float) -> float:
    return block_x + ICON_WIDTH + THETA_TO_NETWORK_GAP


def network_box(block_x: float, group: Group) -> Box:
    return Box(network_x(block_x), group.main_centre_y - NETWORK_HEIGHT / 2, ICON_WIDTH, NETWORK_HEIGHT)


def output_box(block_x: float, group: Group) -> Box:
    return Box(network_x(block_x) + COLUMN_PITCH, group.main_y, ICON_WIDTH, ICON_HEIGHT)


def receiver_box(block_x: float, group: Group) -> Box:
    return Box(network_x(block_x) + 2 * COLUMN_PITCH, group.main_y, ICON_WIDTH, ICON_HEIGHT)


def sender_box(block_x: float, group: Group) -> Box:
    return Box(network_x(block_x) + 2 * COLUMN_PITCH, group.alice_y, ICON_WIDTH, ICON_HEIGHT)


def update_box(theta: Box, group: Group) -> Box:
    return Box(theta.x, group.update_centre_y - UPDATE_HEIGHT / 2, ICON_WIDTH, UPDATE_HEIGHT)


def data_box(group: Group) -> Box:
    return Box(FIRST_THETA_X, group.alice_y, ICON_WIDTH, ICON_HEIGHT)


def step_box(column: int) -> Box:
    start, end = chain_positions(column)
    left = theta_x(start) + ICON_WIDTH + STEP_BOX_LEFT_OFFSET
    right = theta_x(end) + ICON_WIDTH + STEP_BOX_OVERHANG
    return Box(left, STEP_BOX_TOP_Y, right - left, FIGURE_HEIGHT - 2 * STEP_BOX_TOP_Y)


def lane_right() -> float:
    return theta_x(4) + ICON_WIDTH + LANE_RIGHT_MARGIN


# ---------- drawing ----------


def add_plot_frame(layers: Layers, box: Box) -> None:
    layers.add("plot frames", rect(box, FRAME, 0.15))


def draw_lanes(layers: Layers, group: Group, variable_name: str) -> None:
    width = lane_right() - LANE_LEFT_X
    alice = Box(LANE_LEFT_X, group.top, width, ALICE_LANE_HEIGHT)
    bob = Box(LANE_LEFT_X, alice.bottom + LANE_SPACING, width, GROUP_HEIGHT - ALICE_LANE_HEIGHT - LANE_SPACING)
    for name, box in (("Alice", alice), ("Bob", bob)):
        layers.add("lanes and boxes", rect(box, FRAME, 0.3, corner_radius=1.5))
        add_label(layers, LANE_LEFT_X - 1.7, box.centre_y, name, rotate=True)
    add_label(layers, lane_right() + VARIABLE_LABEL_OFFSET, group.top + GROUP_HEIGHT / 2, variable_name, rotate=True)


def draw_step_boxes(layers: Layers) -> None:
    layers.add("background", rect(Box(0, 0, FIGURE_WIDTH, FIGURE_HEIGHT), "none", 0, fill=BACKGROUND))
    for column, step in enumerate(SHOWN_STEPS):
        box = step_box(column)
        layers.add("steps", rect(box, STEP_OUTLINE, 0.35, corner_radius=2.0, fill=STEP_FILL))
        label = "n" if step == STEP_COUNT else str(step)
        add_label(layers, box.centre_x, HEADER_Y, f"i = {label}")


def draw_chain_thetas(layers: Layers, group: Group) -> None:
    for position, name in enumerate(INPUT_NAMES):
        box = theta_box(position, group)
        add_plot_frame(layers, box)
        if position in POSITIONS_FED_BY_UPDATE:
            add_label(layers, box.x + UPDATE_ARROW_INSET + LABEL_SIDE_OFFSET, box.y - LABEL_GAP, name, anchor="start")
        else:
            add_label(layers, box.centre_x, box.y - LABEL_GAP, name)
    ellipsis_start, ellipsis_end = theta_box(2, group).right, theta_box(3, group).x
    y = group.main_centre_y
    add_arrow(layers, [(ellipsis_start, y), (ellipsis_start + ELLIPSIS_ARROW_LENGTH, y)])
    add_label(layers, (ellipsis_start + ellipsis_end) / 2, y + 0.6, "⋯")
    add_arrow(layers, [(ellipsis_end - ELLIPSIS_ARROW_LENGTH, y), (ellipsis_end, y)])


def draw_data_bus(layers: Layers, group: Group) -> None:
    box = data_box(group)
    add_plot_frame(layers, box)
    add_label(layers, box.centre_x, box.y - LABEL_GAP, "x")
    last_sender = sender_box(theta_x(3), group)
    corner_x = box.right + 1.0
    add_line(layers, [(box.right, box.centre_y), (corner_x, box.centre_y), (corner_x, group.bus_y), (last_sender.centre_x, group.bus_y)])


def draw_step(layers: Layers, column: int, group: Group, is_labelled: bool) -> None:
    start, end = chain_positions(column)
    block_x = theta_x(start)
    previous_theta, next_theta = theta_box(start, group), theta_box(end, group)
    network, output = network_box(block_x, group), output_box(block_x, group)
    receiver, sender = receiver_box(block_x, group), sender_box(block_x, group)
    update = update_box(next_theta, group)
    for box in (output, receiver, sender):
        add_plot_frame(layers, box)
    add_boxed_label(layers, network, "Network")
    add_boxed_label(layers, update, "update")

    main_y = group.main_centre_y
    junction_x = previous_theta.right + JUNCTION_OFFSET
    add_arrow(layers, [(previous_theta.right, main_y), (network.x, main_y)])
    add_junction(layers, junction_x, main_y)
    add_arrow(layers, [(junction_x, main_y), (junction_x, group.update_centre_y), (update.x, group.update_centre_y)])
    add_arrow(layers, [(network.right, main_y), (output.x, main_y)])
    add_arrow(layers, [(output.right, main_y), (receiver.x, main_y)])
    add_arrow(layers, [(sender.centre_x, group.bus_y), (sender.centre_x, sender.y)])
    add_arrow(layers, [(sender.right, sender.centre_y), (update.centre_x, sender.centre_y), (update.centre_x, update.y)])
    update_arrow_x = next_theta.x + UPDATE_ARROW_INSET
    add_arrow(layers, [(update_arrow_x, update.bottom), (update_arrow_x, next_theta.y)])
    kl_x = receiver.x + receiver.width * KL_ARROW_POSITION
    add_arrow(layers, [(kl_x, receiver.y), (kl_x, sender.bottom)], dashed=True)

    add_label(layers, kl_x + LABEL_SIDE_OFFSET, group.top + KL_LABEL_OFFSET, "KL", anchor="start", size=SMALL_FONT_SIZE)
    if is_labelled:
        add_label(layers, update.centre_x + LABEL_SIDE_OFFSET, group.top + SAMPLE_LABEL_OFFSET, "y", anchor="start", size=SMALL_FONT_SIZE)
        add_label(layers, sender.x - SIDE_LABEL_CLEARANCE, sender.centre_y + 0.8, "p_S", anchor="end")
        add_label(layers, output.centre_x, output.y - LABEL_GAP, "p_O")
        add_label(layers, kl_x - LABEL_SIDE_OFFSET, receiver.y - LABEL_GAP, "p_R", anchor="end")


def draw_network_links(layers: Layers) -> None:
    for column in range(len(SHOWN_STEPS)):
        block_x = theta_x(chain_positions(column)[0])
        upper, lower = network_box(block_x, CONTINUOUS_GROUP), network_box(block_x, DISCRETE_GROUP)
        layers.add("network link", polyline([(upper.centre_x, upper.bottom), (lower.centre_x, lower.y)], FRAME, LINE_WIDTH, dashed=True))


def draw_continuous_icons(layers: Layers, column: int, step: ContinuousStep) -> None:
    group = CONTINUOUS_GROUP
    start, end = chain_positions(column)
    block_x = theta_x(start)
    deviation = 1 / math.sqrt(step.accuracy)
    previous, current = theta_box(start, group), theta_box(end, group)
    sender, output, receiver = sender_box(block_x, group), output_box(block_x, group), receiver_box(block_x, group)
    layers.add("plots", gaussian_icon(sender, DATA_VALUE, deviation, TRUTH_COLOR))
    layers.add("plots", delta_icon(output, step.prediction, BELIEF_COLOR))
    layers.add("plots", gaussian_icon(receiver, step.prediction, deviation, BELIEF_COLOR))
    if start in (0, 3):
        layers.add("plots", gaussian_icon(previous, step.previous_mean, 1 / math.sqrt(step.previous_precision), BELIEF_COLOR))
    layers.add("plots", gaussian_icon(current, step.mean, 1 / math.sqrt(step.precision), BELIEF_COLOR))
    for box in (output, receiver, previous, current):
        layers.add("reference lines", reference_line(box))
    for box in (sender, output, receiver, previous, current):
        add_value_axis(layers, box, is_labelled=column == 0)


def draw_discrete_icons(layers: Layers, column: int, step: DiscreteStep) -> None:
    group = DISCRETE_GROUP
    start, end = chain_positions(column)
    block_x = theta_x(start)
    sender, output, receiver = sender_box(block_x, group), output_box(block_x, group), receiver_box(block_x, group)
    elements = logit_slot_icons(sender, step.accuracy, one_hot(TRUE_CLASS))
    elements += bar_icons(output, step.output)
    elements += logit_slot_icons(receiver, step.accuracy, step.output)
    if start in (0, 3):
        elements += bar_icons(theta_box(start, group), step.previous_probabilities)
    elements += bar_icons(theta_box(end, group), step.probabilities)
    for element in elements:
        layers.add("plots", element)
    for box in (output, theta_box(start, group), theta_box(end, group)):
        add_class_axis(layers, box, is_labelled=column == 0)
    add_logit_axes(layers, sender, is_labelled=column == 0, has_y_numbers=True)
    add_logit_axes(layers, receiver, is_labelled=column == 0, has_y_numbers=False)


def draw_data_icons(layers: Layers) -> None:
    continuous_data, discrete_data = data_box(CONTINUOUS_GROUP), data_box(DISCRETE_GROUP)
    layers.add("plots", delta_icon(continuous_data, DATA_VALUE, TRUTH_COLOR))
    for element in bar_icons(discrete_data, one_hot(TRUE_CLASS)):
        layers.add("plots", element)
    add_value_axis(layers, continuous_data, is_labelled=True)
    add_class_axis(layers, discrete_data, is_labelled=True)


def build_figure() -> Layers:
    rng = random.Random(SEED)
    continuous_steps = run_continuous_chain(rng)
    discrete_steps = run_discrete_chain(rng)
    layers = Layers()
    draw_step_boxes(layers)
    for group, name in ((CONTINUOUS_GROUP, "Variable 1 (continuous)"), (DISCRETE_GROUP, "Variable 2 (discrete)")):
        draw_lanes(layers, group, name)
        draw_chain_thetas(layers, group)
        draw_data_bus(layers, group)
    draw_data_icons(layers)
    draw_network_links(layers)
    for column, step in enumerate(SHOWN_STEPS):
        for group in (CONTINUOUS_GROUP, DISCRETE_GROUP):
            draw_step(layers, column, group, is_labelled=column == 0)
        draw_continuous_icons(layers, column, continuous_steps[step - 1])
        draw_discrete_icons(layers, column, discrete_steps[step - 1])
    return layers


def main() -> None:
    OUTPUT_PATH.write_text(build_figure().to_svg(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
