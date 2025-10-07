# %%
import turtle
from typing import Dict, List, Callable, Tuple, Optional

def generate_lsystem(axiom: str, rules: Dict[str, str], iterations: int) -> List[str]:
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(ch, ch) for ch in s)
    return list(s)

def render_lsystem(
    seq: List[str],
    angle: float,
    step: float,
    *,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: int = 0,
    custom_actions: Optional[Dict[str, Callable[[turtle.Turtle, float, float, list], None]]] = None,
    window_title: str = "L-System"
) -> None:
    screen = turtle.Screen()
    screen.title(window_title)
    pen = turtle.Turtle(visible=False)
    pen.speed(speed)
    pen.penup()
    pen.setheading(heading)
    pen.goto(*start_pos)
    pen.pendown()

    stack: list = []

    def act_forward(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.forward(st)

    def act_move(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.penup(); p.forward(st); p.pendown()

    def act_left(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.left(a)

    def act_right(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.right(a)

    def act_push(p: turtle.Turtle, a: float, st: float, _stack: list):
        _stack.append((p.position(), p.heading()))

    def act_pop(p: turtle.Turtle, a: float, st: float, _stack: list):
        if _stack:
            pos, hd = _stack.pop()
            p.penup(); p.goto(pos); p.setheading(hd); p.pendown()

    default_actions: Dict[str, Callable[[turtle.Turtle, float, float, list], None]] = {
        "F": act_forward,
        "G": act_forward,
        "f": act_move,
        "+": act_left,
        "-": act_right,
        "[": act_push,
        "]": act_pop,
    }

    if custom_actions:
        default_actions.update(custom_actions)

    for ch in seq:
        action = default_actions.get(ch)
        if action:
            action(pen, angle, step, stack)

    turtle.done()

seq = generate_lsystem("X", {"X": "F-[[X]+X]+F[+FX]-X", "F": "FF"}, iterations=4)
render_lsystem(seq, angle=24, step=5, start_pos=(0, -300), heading=90, speed=0, window_title="Plant")
# %%
