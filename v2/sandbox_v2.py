from concurrent.futures import ProcessPoolExecutor
import ast
from enum import Enum
from threading import TIMEOUT_MAX
from uu import Error

"""

"""


class Status(Enum):
    OK = "OK"
    TIMEOUT = "TIMEOUT"
    OPLIMIT = "OPLIMIT"
    MEM = "MEM"
    RECURSION = "RECURSION"
    FORBIDDEN = "FORBIDDEN"
    ERROR_NO_MESSAGE = "ERROR_NO_MESSAGE"
    ERROR = "ERROR"

class ValidationError(Error):
    pass




def prepare(
    source: str,
    *,
    mode: str = "allow",           # "allow" or "ban"
    op_limit: int = 10_000,
    instrument: bool = True
) -> str:
    """
    Parse + validate `source` according to `mode`, then instrument it for op-budget if requested.
    Return the transformed source as a string or raise a ValidationError (FORBIDDEN case).
    """
    try:
        ast.parse(source)
    except SyntaxError:
        raise ValidationError("Not valid AST")

    

def run_single(
    prepared_source: str,
    entry: str,
    args: tuple = (),
    *,
    timeout_s: float = 1.0,
    limits: dict | None = None,    # e.g., {"cpu_s":1, "as_bytes": 256*1024*1024, "nofile":64}
) -> dict:
    """
    Execute a prepared source in a separate process.
    Return a dict with keys: status, result, runtime_ms, stdout, stderr.
    Must kill on timeout and never leave a zombie.
    """

def run_batch(
    cases: list[dict],
    *,
    mode: str = "allow",
    op_limit: int = 10_000,
    instrument: bool = True,
    timeout_s: float = 1.0,
    limits: dict | None = None,
    max_workers: int = 4
) -> dict:
    """
    For each case: {"src": str, "func": str, "args": list}, prepare + run in parallel.
    Return {"results": [ {status, result, runtime_ms, stdout, stderr}, ... ]} in the same order as input.
    Preparation failures should yield status="FORBIDDEN" for that case.
    """
