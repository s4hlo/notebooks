# %%
import random
import turtle
from typing import Dict, List, Callable, Tuple, Optional


def generate_lsystem(axiom: str, rules: Dict[str, str], iterations: int) -> List[str]:
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(ch, ch) for ch in s)
    return list(s)


def generate_stochastic_lsystem(
    axiom: str, rules: Dict[str, List[Tuple[str, float]]], iterations: int, seed: int
) -> List[str]:
    random.seed(seed)
    s = axiom
    for _ in range(iterations):
        s = "".join(
            (
                random.choices(
                    [r[0] for r in rules[ch]], weights=[r[1] for r in rules[ch]], k=1
                )[0]
                if ch in rules
                else ch
            )
            for ch in s
        )
    return list(s)


def render_l_system_iteractive(seq: List[str]):
    screen = turtle.Screen()
    screen.title("L-System")
    screen.bgcolor("gray")
    pen = turtle.Turtle(visible=False)
    pen.speed(0)
    pen.pensize(3)
    pen.penup()
    pen.setheading(90)
    pen.goto(0, 0)
    pen.pendown()


def render_lsystem(
    seq: List[str],
    angle: float,
    step: float,
    *,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: int = 0,
) -> None:

    branches = "#ca9ee6"
    leaves = "#a6d189"
    background = "#303446"

    screen = turtle.Screen()
    screen.bgcolor(background)
    pen = turtle.Turtle(visible=False)
    pen.speed(speed)
    pen.pensize(3)
    pen.penup()
    pen.setheading(heading)
    pen.goto(*start_pos)
    pen.pendown()

    stack: list = []

    def act_forward(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.color(branches)
        p.forward(st)

    def act_forward_leaf(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.color(leaves)
        p.forward(st)

    def act_move(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.penup()
        p.forward(st)
        p.pendown()

    def act_left(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.left(a)

    def act_right(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.right(a)

    def act_push(p: turtle.Turtle, a: float, st: float, _stack: list):
        _stack.append((p.position(), p.heading()))

    def act_pop(p: turtle.Turtle, a: float, st: float, _stack: list):
        if _stack:
            pos, hd = _stack.pop()
            p.penup()
            p.goto(pos)
            p.setheading(hd)
            p.pendown()

    default_actions: Dict[str, Callable[[turtle.Turtle, float, float, list], None]] = {
        "L": act_forward_leaf,
        "F": act_forward,
        "G": act_forward,
        "f": act_move,
        "+": act_left,
        "-": act_right,
        "[": act_push,
        "]": act_pop,
    }

    for ch in seq:
        action = default_actions.get(ch)
        if action:
            action(pen, angle, step, stack)

    turtle.done()


# %%


wiki_plant_stochastic = {
    "axiom": "-X",
    "rules": {
        "X": [("F-[[X]+X]+F[+FX]-X", 0.1), ("F+[[X]-X]-F[-FX]+X", 0.9)],
        "F": [("FF", 1.0)],
    },
    "angle": 25,
}

sierpinski_triangle = {
    "axiom": "F-G-G",
    "rules": {
        "F": "F-G+F+G-F",
        "G": "GG",
    },
    "angle": 120,
}

wiki_plant = {
    "axiom": "-X",
    "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
    "angle": 25,
}

wiki_plant_with_leaves = {
    "axiom": "-X",
    "rules": {"X": "F+[[X]-X]-F[-FX]+XL", "F": "FF", "L": "L[-L+L]+[+L-L]"},
    "angle": 25,
}


fractal_plant = {
    "axiom": "X",
    "rules": {"X": "F[-X][+X]FXL", "F": "FF", "L": "L[-L+L]+[+L-L]"},
    "angle": 25,
}


basic_leaf = {
    "axiom": "L",
    "rules": {
        # Corpo denso e simétrico: veias curtas expandem e depois recolhem
        "L": "LL-[-L+L+L]+[+L-L-L]"
    },
    "angle": 20,  # ângulo menor deixa a folha mais arredondada
}

iterations = 5
seed = 42
stochastic = False
cfg = wiki_plant_with_leaves


generate_func = generate_stochastic_lsystem if stochastic else generate_lsystem
args = (cfg["axiom"], cfg["rules"], iterations) + ((seed,) if stochastic else ())
sequence = generate_func(*args)
angle_cfg = cfg["angle"]
render_lsystem(
    sequence,
    angle=angle_cfg,
    step=8,
    start_pos=(0, -300),
    heading=90,
    speed=0,
)

# %%
