# %%
import turtle
from typing import Dict, List, Callable, Tuple, Optional


def generate_lsystem(axiom: str, rules: Dict[str, str], iterations: int) -> List[str]:
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(ch, ch) for ch in s)
    return list(s)


def generate_stochastic_lsystem(
    axiom: str, rules: Dict[str, str], iterations: int, seed: int
) -> List[str]:
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
    custom_actions: Optional[
        Dict[str, Callable[[turtle.Turtle, float, float, list], None]]
    ] = None,
    window_title: str = "L-System",
    color1: str = "blue",
    color2: str = "red",
) -> None:
    screen = turtle.Screen()
    screen.title(window_title)
    screen.bgcolor("gray")
    pen = turtle.Turtle(visible=False)
    pen.speed(speed)
    pen.pensize(3)
    pen.penup()
    pen.setheading(heading)
    pen.goto(*start_pos)
    pen.pendown()

    stack: list = []
    current_color = color1

    def act_forward(p: turtle.Turtle, a: float, st: float, _stack: list):
        p.color(current_color)
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
        nonlocal current_color
        current_color = color2 if current_color == color1 else color1
        _stack.append((p.position(), p.heading()))

    def act_pop(p: turtle.Turtle, a: float, st: float, _stack: list):
        if _stack:
            nonlocal current_color
            current_color = color2 if current_color == color1 else color1
            pos, hd = _stack.pop()
            p.penup()
            p.goto(pos)
            p.setheading(hd)
            p.pendown()

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


i = 3

L_SYSTEMS = {
    # --- Curvas e fractais clássicos ---
    "koch_curve": {"axiom": "F", "rules": {"F": "F+F--F+F"}, "angle": 60},
    "koch_snowflake": {"axiom": "F--F--F", "rules": {"F": "F+F--F+F"}, "angle": 60},
    "levy_c": {"axiom": "F", "rules": {"F": "+F--F+"}, "angle": 45},
    "heighway_dragon": {
        "axiom": "FX",
        "rules": {"X": "X+YF+", "Y": "-FX-Y"},
        "angle": 90,
    },
    "hilbert_curve": {
        "axiom": "A",
        "rules": {"A": "-BF+AFA+FB-", "B": "+AF-BFB-FA+"},
        "angle": 90,
    },
    "moore_curve": {
        "axiom": "LFL+F+LFL",
        "rules": {"L": "-RF+LFL+FR-", "R": "+LF-RFR-FL+"},
        "angle": 90,
    },
    "peano_curve": {
        "axiom": "X",
        "rules": {"X": "XFYFX+F+YFXFY-F-XFYFX", "Y": "YFXFY-F-XFYFX+F+YFXFY"},
        "angle": 90,
    },
    "gosper_curve": {
        "axiom": "A",
        "rules": {"A": "A+B++B-A--AA-B+", "B": "-A+BB++B+A--A-B"},
        "angle": 60,
    },
    "sierpinski_arrowhead": {
        "axiom": "A",
        "rules": {"A": "B-A-B", "B": "A+B+A"},
        "angle": 60,
    },
    "sierpinski_triangle": {
        "axiom": "F-G-G",
        "rules": {"F": "F-G+F+G-F", "G": "GG"},
        "angle": 120,
    },
    "quadratic_koch_island": {
        "axiom": "F+F+F+F",
        "rules": {"F": "F+F-F+F+F"},
        "angle": 90,
    },
    "minkowski_sausage": {"axiom": "F", "rules": {"F": "F+F-F-FF+F+F-F"}, "angle": 90},
    "terdragon_curve": {"axiom": "F", "rules": {"F": "F+F-F"}, "angle": 120},
    "vicsek_fractal": {"axiom": "F", "rules": {"F": "F+F+F+F+F"}, "angle": 90},
    "harter_square": {"axiom": "F+F+F+F", "rules": {"F": "FF+F+F+F+FF"}, "angle": 90},
    # --- Plantas e árvores ---
    "binary_tree": {"axiom": "F", "rules": {"F": "F[+F]F[-F]F"}, "angle": 25},
    "wiki_plant": {
        "axiom": "-X",
        "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
        "angle": 25,
    },
    "prusinkiewicz_plant": {
        "axiom": "X",
        "rules": {"X": "F-[[X]+X]+F[+FX]-X", "F": "FF"},
        "angle": 25,
    },
    "bush_symmetric": {
        "axiom": "F",
        "rules": {"F": "FF-[-F+F+F]+[+F-F-F]"},
        "angle": 20,
    },
    "asymmetric_tree": {"axiom": "F", "rules": {"F": "F[+F]F[-F][F]"}, "angle": 22.5},
    "short_internodes_plant": {
        "axiom": "X",
        "rules": {"X": "F[+X]F[-X]+X", "F": "FF", "f": "f"},
        "angle": 22.5,
    },
    # --- Educativos e não geométricos ---
    "fibonacci_word": {"axiom": "A", "rules": {"A": "AB", "B": "A"}, "angle": 0},
    "cantor_string": {"axiom": "A", "rules": {"A": "ABA", "B": "BBB"}, "angle": 0},
    # --- Suas variantes ---
    "my": {
        "axiom": "A",
        "rules": {"A": "+BfFf-[AFA]-FffB+", "B": "-AF+BFB+FA-"},
        "angle": 90,
    },
    "hilbert_variant": {
        "axiom": "A",
        "rules": {"A": "+BF-AFA-FB+", "B": "-AF+BFB+FA-"},
        "angle": 90,
    },
    "dragon_curve": {"axiom": "FX", "rules": {"X": "X+YF+", "Y": "-FX-Y"}, "angle": 90},
    "real_binary_tree": {"axiom": "F", "rules": {"F": "F[-F][+F]"}, "angle": 20},
    "real_direct_binary_tree": {
        "axiom": "A",
        "rules": {"A": "F[+A][-A]A", "F": "F"},
        "angle": 45,
    },
}
key_list = list(L_SYSTEMS.keys())


def runner(name: str, iterations: int = 6):
    cfg = L_SYSTEMS[name]
    sequence = generate_lsystem(cfg["axiom"], cfg["rules"], iterations=iterations)
    angle_cfg = cfg["angle"]
    render_lsystem(
        sequence,
        angle=angle_cfg,
        step=32,
        start_pos=(0, 0),
        heading=90,
        speed=0,
        window_title=f"Beautiful Plant",
        color1="green",
        color2="darkgreen",
    )
# %%
runner("asymmetric_tree", 3)


# %%
