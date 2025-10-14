
from __future__ import annotations
import os, math, hashlib, random
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, List, Tuple, TypeVar, Optional
from collections import defaultdict

T = TypeVar("T")
Outcome = int

class Once(Generic[T]):
    __slots__ = ("_value","_used")
    def __init__(self, value: T):
        self._value = value
        self._used = False
    def consume(self) -> T:
        if self._used:
            raise RuntimeError("Once-token already consumed")
        self._used = True
        v, self._value = self._value, None
        return v

def derive_seed(master_seed: bytes, label: str) -> float:
    import hashlib, math
    h = hashlib.sha256(master_seed + label.encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big") / 2**64
    return 2.0 * math.pi * x

@dataclass(frozen=True)
class RClosure:
    seed: float
    interp: Callable[[float, float, float], Outcome]
    def measure(self, context: float, local_noise: float) -> Outcome:
        return self.interp(context, self.seed, local_noise)

def spin_interp(context_angle: float, seed: float, local_noise: float, *, flip: bool=False) -> Outcome:
    import math
    p = (1.0 + math.cos(context_angle - seed)) / 2.0
    if flip: p = 1.0 - p
    return +1 if local_noise < p else -1

def make_spin_closure(master_seed: bytes, *, flip: bool=False) -> RClosure:
    seed = derive_seed(master_seed, "spin")
    return RClosure(seed=seed, interp=lambda ctx, s, r: spin_interp(ctx, s, r, flip=flip))

class Super(Generic[T]):
    def __init__(self, terms: Optional[Dict[T, complex]] = None):
        self.amp: Dict[T, complex] = dict(terms) if terms else {}
    @staticmethod
    def pure(state: T) -> "Super[T]":
        return Super({state: 1+0j})
    def norm2(self) -> float:
        return float(sum((abs(a)**2 for a in self.amp.values())))
    def normalize(self) -> "Super[T]":
        n2 = self.norm2()
        if n2 == 0.0: raise ValueError("Zero vector cannot be normalized.")
        s = 1.0 / math.sqrt(n2)
        self.amp = {k: a*s for k,a in self.amp.items()}
        return self
    def map_linear(self, f: Callable[[T], Iterable[Tuple[T, complex]]]) -> "Super[T]":
        from collections import defaultdict
        acc: Dict[T, complex] = defaultdict(complex)
        for s, a in self.amp.items():
            for s2, w in f(s):
                acc[s2] += a * w
        return Super(acc)
    def probs(self) -> Dict[T, float]:
        return {s: float(abs(a)**2) for s, a in self.amp.items()}
    def measure(self, rng: random.Random) -> Tuple[T, "Super[T]"]:
        items = list(self.amp.items())
        states = [s for s,_ in items]
        weights = [abs(a)**2 for _,a in items]
        total = sum(weights)
        if total == 0.0: raise ValueError("Cannot measure zero-amplitude state.")
        r = rng.random() * total
        cum = 0.0
        for s, w in zip(states, weights):
            cum += w
            if r <= cum:
                return s, Super.pure(s)
        return states[-1], Super.pure(states[-1])

INV_SQRT2 = 1.0 / math.sqrt(2.0)
def H_map(bit: str):
    if bit == "0":
        yield ("0", INV_SQRT2+0j); yield ("1", INV_SQRT2+0j)
    elif bit == "1":
        yield ("0", INV_SQRT2+0j); yield ("1", -INV_SQRT2+0j)
    else:
        raise ValueError('Bit must be "0" or "1"')



# quick demo
def _demo():
    print("=== R-Closure demo ===")
    master = os.urandom(16)
    A = make_spin_closure(master, flip=False)
    B = make_spin_closure(master, flip=True)
    rngA, rngB = random.Random(7), random.Random(11)
    a, b = 0.0, math.pi/3
    trials, corr = 3000, 0
    for _ in range(trials):
        corr += A.measure(a, rngA.random()) * B.measure(b, rngB.random())
    print(f"E[A*B] at a=0, b=π/3 ≈ {corr/trials:.3f}")
    print("\n=== Superposition demo ===")
    rng = random.Random(42)
    psi = Super.pure("0").map_linear(H_map)  # H|0>
    print("Amplitudes after H:", {k: complex(v) for k,v in psi.amp.items()})
    outcome, collapsed = psi.measure(rng)
    print("Outcome:", outcome, "Collapsed:", collapsed.amp)

_demo()
