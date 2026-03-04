from dataclasses import dataclass
import sys

@dataclass
class Behavior:
    id: str
    vf: float
    af: float
    vt: float
    at: float
    th: float

SPEEDUP = 1.0 # ← adjust this factor (e.g., 1.0 = original, 1.1 = +10%, 0.9 = -10%)

BEHAVIORS = {
    "C1":  Behavior(id="C1",  vf=(5.0/100) * SPEEDUP,         af=-1.385 , vt=(8.889333333/100) * SPEEDUP, at=1.506 , th=0.6),
    "C2":  Behavior(id="C2",  vf=(5.012666667/100) * SPEEDUP, af=1.4 ,    vt=(7.196666667/100) * SPEEDUP, at=-0.4 ,  th=0.6),
    "C3":  Behavior(id="C3",  vf=(5.0/100) * SPEEDUP,         af=-1.026 , vt=(5.0/100) * SPEEDUP,        at=0.585 , th=0.6),
    "D1":  Behavior(id="D1",  vf=(5.364333333/100) * SPEEDUP, af=-1.026 , vt=(-5.929333333/100) * SPEEDUP,at=-0.48 , th=0.6),
    "D2":  Behavior(id="D2",  vf=(5.556666667/100) * SPEEDUP, af=-1.392 , vt=(-6.078333333/100) * SPEEDUP,at=-0.61 , th=0.6),
    "D3":  Behavior(id="D3",  vf=(-5.196666667/100) * SPEEDUP,af=-1.548 , vt=(-8.466666667/100) * SPEEDUP,at=-0.561 ,th=0.6),
    "A1": Behavior(id="A1", vf=(-8.636666667/100) * SPEEDUP,af=-0.86 ,  vt=(5.0/100) * SPEEDUP,        at=-1.567 ,th=0.6),
    "A2": Behavior(id="A2", vf=(-8.073666667/100) * SPEEDUP,af=-0.475 , vt=(-5.945666667/100) * SPEEDUP,at=-0.889 ,th=0.6),
    "A3": Behavior(id="A3", vf=(-9.0/100) * SPEEDUP,        af=0.4 ,    vt=(7.393/100) * SPEEDUP,      at=0.4 ,   th=0.6)
}

# Helper to get behavior by id
def get_behavior_by_id(behavior_id: str) -> Behavior:
    for behavior in BEHAVIORS:
        if behavior.id == behavior_id:
            return behavior
    raise ValueError(f"Behavior with id '{behavior_id}' not found.")

# Example usage for shell scripts:
# python3 known_controllers.py BHV002
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 known_controllers.py <BEHAVIOR_ID>")
        exit(1)
    behavior_id = sys.argv[1]
    try:
        behavior = get_behavior_by_id(behavior_id)
        print(f"{behavior.id} {behavior.vt} {behavior.at} {behavior.vf} {behavior.af} {behavior.th}")
    except ValueError as e:
        print(e)
        exit(1)