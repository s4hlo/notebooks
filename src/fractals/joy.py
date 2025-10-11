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
    screen: turtle.Screen,
    seq: List[str],
    angle: float,
    step: float,
    *,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: int = 0,
) -> None:

    branches = "#8B4513"  # Marrom
    leaves = "#FF69B4"  # Rosa
    screen.tracer(10)  # Atualiza a cada 10 movimentos (bem rápido)
    pen = turtle.Turtle(visible=False)
    pen.speed(0)  # Velocidade máxima
    pen.pensize(4)
    pen.penup()
    pen.setheading(heading)
    pen.goto(*start_pos)
    pen.pendown()

    stack: list = []
    current_pensize = 10
    current_step = step
    leave_pensize = 4

    def thick_reduction(current_pensize: int):
        # return current_pensize
        return max(1, current_pensize * (0.95 - 0.02 * (current_pensize**0.5)))

    def step_reduction(current_step: int):
        return current_step
        return max(1, current_step * (0.95 - 0.02 * (current_step**0.5)))

    def act_forward(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.color(branches)
        p.pensize(current_pensize)
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
        nonlocal current_pensize
        nonlocal current_step
        _stack.append((p.position(), p.heading(), current_pensize, current_step))
        current_pensize = thick_reduction(current_pensize)
        current_step = step_reduction(current_step)

    def act_pop(p: turtle.Turtle, a: float, st: float, _stack: list):
        if _stack:
            nonlocal current_pensize
            nonlocal current_step
            pos, hd, old_pensize, old_step = _stack.pop()
            current_pensize = thick_reduction(old_pensize)
            current_step = step_reduction(old_step)
            p.penup()
            p.goto(pos)
            p.setheading(hd)
            p.pendown()

    def draw_leaf(p: turtle.Turtle, a: float, st: float, _stack: list):
        act_push(p, a, st, _stack)
        p.pensize(leave_pensize)
        p.color(leaves)
        p.right(45)
        p.forward(st)
        p.left(60)
        p.forward(st)
        p.left(120)
        p.forward(st)
        p.left(60)
        p.forward(st)
        p.pensize(current_pensize)
        act_pop(p, a, st, _stack)

    default_actions: Dict[str, Callable[[turtle.Turtle, float, float, list], None]] = {
        "L": draw_leaf,
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
            action(pen, angle, current_step, stack)


def details_background(screen: turtle.Screen):
    pen = turtle.Turtle(visible=False)
    pen.speed(0)
    pen.pensize(4)
    pen.penup()
    pen.setheading(90)
    pen.goto(400, 280)
    pen.pendown()

    # draw a sun on the right top
    pen.color("#FFD700")
    pen.begin_fill()
    pen.circle(100)
    pen.end_fill()
    pen.penup()
    pen.goto(100, 100)
    pen.pendown()

    # add randow stars
    for _ in range(100):
        pen.color("#FFFFFF")
        pen.pensize(1)
        pen.begin_fill()
        pen.circle(1)
        pen.end_fill()
        pen.penup()
        pen.goto(random.randint(-500, 500), random.randint(-500, 500))
        pen.pendown()


wiki_plant_with_leaves = {
    "axiom": "-X",
    "rules": {"X": "F+[[X]-XL]-F[-FX]+XL", "F": "FF"},
    "angle": 25,
}


iterations = 6
seed = 42
stochastic = False
cfg = wiki_plant_with_leaves


generate_func = generate_stochastic_lsystem if stochastic else generate_lsystem
args = (cfg["axiom"], cfg["rules"], iterations) + ((seed,) if stochastic else ())
sequence = generate_func(*args)
angle_cfg = cfg["angle"]

screen = turtle.Screen()
background = "#303446"
screen.bgcolor(background)
details_background(screen)
render_lsystem(
    screen,
    sequence,
    angle=angle_cfg,
    step=4,
    start_pos=(-300, -300),
    heading=90,
    speed=0,
)

screen.update()  # Atualiza a tela final
turtle.done()

# %%
