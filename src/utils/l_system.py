from typing import Dict, List, Callable, Tuple, Optional

def generate_lsystem(axiom: str, rules: Dict[str, str], iterations: int) -> List[str]:
    s = axiom
    for _ in range(iterations):
        s = "".join(rules.get(ch, ch) for ch in s)
    return list(s)