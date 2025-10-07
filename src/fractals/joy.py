# %%
import turtle
from typing import Dict, List, Callable, Tuple, Optional
from src.fractals.l_system import generate_lsystem


def render_lsystem(
    seq: List[str],
    angle: float,
    step: float,
    *,
    start_pos: Tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: int = 0,
    custom_actions: Optional[Dict[str, Callable[[turtle.Turtle, float, float, list], None]]] = None,
    window_title: str = "L-System",
    color1: str = "blue",
    color2: str = "red"
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
    current_color = color1

    def act_forward(p: turtle.Turtle, a: float, st: float, _stack: list):
        nonlocal current_color
        p.color(current_color)
        p.forward(st)
        current_color = color2 if current_color == color1 else color1

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

def render_iterations_separate(sequences: List[List[str]], angle: float, step: float, spacing: float = 200):
    """Desenha cada sequência em posições separadas, sem sobreposição"""
    for i, seq in enumerate(sequences):
        start_x = i * spacing - (len(sequences) - 1) * spacing / 2  # Centralizar
        render_lsystem(seq, angle=angle, step=step, 
                      start_pos=(start_x, -300), heading=90, speed=0, 
                      window_title=f"Iteração {i}")

# Exemplo de uso
sequences = []
for i in range(4):
    seq = generate_lsystem("F", {"F": "F-F++F-F"}, iterations=i)
    sequences.append(seq)

render_iterations_separate(sequences, angle=60, step=8, spacing=150)
# %%
