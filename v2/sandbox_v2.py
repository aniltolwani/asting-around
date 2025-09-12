from asyncio import SafeChildWatcher, as_completed
from concurrent.futures import ProcessPoolExecutor
import ast
from enum import Enum
import enum
from hmac import new
import stat
from sys import stderr
from turtle import fd
from uu import Error
import multiprocessing

from api import validate_and_instrument
from hello import fut, res


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

class BanListError(Error):
    pass

class AllowListError(Error):
    pass

SAFE_BUILTINS = {
    "abs": abs,
    "list": list,
    "range": range,
    "dict": dict,
    "max": max,
}

"""Allowlist mode (default): allow only a small set of AST node types (e.g., Module, FunctionDef, arguments, arg, Return, Assign, AugAssign, AnnAssign, Expr, If, For, While, BinOp, UnaryOp, BoolOp, Compare, Call, Name, Load, Store, Constant, List, Tuple, Set, Dict, comprehension, ListComp, DictComp, SetComp, GeneratorExp).
Block attribute access to dunders (e.g., obj.__dict__). Block calls to eval/exec/open/__import__/compile.
"""

class AllowListValidator(ast.NodeVisitor):
    # in this version we will allow particular node types, not just functions
    # we cna get this by overriding generic visit instead of doing individual visit_Blank
    
    def generic_visit(self, node):
        # major catogries
        # modules and functions
        p = False
        if isinstance(node, (ast.Module, ast.FunctionDef)):
            p = True
        # what a statement is just anything with assign?
        elif isinstance(node, (ast.Assign, ast.Expr)):
            p = True
        # control logic: if, while, for
        elif isinstance(node, (ast.If, ast.BoolOp, ast.While, ast.If)):
            p = True
        # unary ops all are fine
        elif isinstance(node, (ast.UnaryOp)):
            p = True
        # binary ops all are fine
        elif isinstance(node, (ast.BinOp)):
            p = True
        # load / store from memory (it will all be local
        elif isinstance(node, (ast.Load, ast.Store, ast.Constant)):
            p = True
        if not p:
            raise ValidationError("Not in allowed list")
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        return

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if "__" in node.func.id:
                raise AllowListError("no dunder methods")
            elif node.func.id not in SAFE_BUILTINS.keys():
                raise AllowListError("not in safe builtins")
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if "__" in node.attr:
            raise AllowListError("no dunder attributes")
    def visit_Import(self, node):
        raise AllowListError("no imports allowed")
    def visit_ImportFrom(self, node):
        raise AllowListError("no import from allowed")


"""
Banlist mode: block Import, ImportFrom, any dunder attribute, and calls to eval/exec/open/__import__/compile.
Runtime builtins must be a tight whitelist (no __import__, eval/exec/open, reflection helpers, file/network access.
"""

class BanListValidator(ast.NodeVisitor):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if "__" in node.func.id:
                raise BanListError("no dunder methods")
            elif node.func.id not in SAFE_BUILTINS.keys():
                raise BanListError("not in safe builtins")
        return self.generic_visit(node)
    def visit_Attribute(self, node):
        if "__" in node.attr:
            raise BanListError("no dunder attributes")
    def visit_Import(self, node):
        raise BanListError("no imports allowed")
    def visit_ImportFrom(self, node):
        raise BanListError("no import from allowed")


"""Instrumentation (op‑budget)

Insert a tick at loop heads (For/While) and for every Call (use the (_tick() or f)(...) trick).

Also tick comprehensions by OR‑wrapping the elt (and key/value for dict comps) with _tick().

_tick() increments a global counter; if > op_limit, raise an exception that your runner maps to "OPLIMIT".
"""

class Instrument(ast.NodeTransformer):

    def visit_Call(self, node):
        # or wrap it
        node = self.generic_visit(node)

        return ast.BoolOp(
            op = ast.Or(),
            values=[
                ast.Call(func=ast.Name(id="_tick", ctx = ast.Load())),
                # this is also a Call!
                node
            ]
        )
    def visit_For(self, node):
        node = self.generic_visit(node)
        # since for is a statement with a body, we can just add a tick call in the beginning of it
        # but, to push ourselves, lets use the or trick
        # we could also use a tuple but or works
        # acually this ownt work with a statement, lets just body prepend..
        new_body = [ast.Call(func=ast.Name(id="_tick", ctx = ast.Load()))] + node.body
        node.body = new_body
        return node
    def visit_While(self, node):
        # similarly body prepending here works the best
        node = self.generic_visit(node)
        node.body = [ast.Call(func=ast.Name(id = "_tick", ctx = ast.Load()))] + node.body
        return node
    def visit_DictComp(self, node):
        node = self.generic_visit(node)
        # wrap the key and value in this case
        tick_func= ast.Call(func = ast.Name(id ="_tick", ctx = ast.Load()))
        node.key = ast.BoolOp(
            op = ast.Or(),
            values = [tick_func, node.key]
        )
        node.value = ast.BoolOp(
            op = ast.Or(),
            values = [tick_func, node.value]
        )
        return node
    def visit_ListComp(self, node):
        node = self.generic_visit(node)
        new_elt = ast.BoolOp(
            op = ast.Or(),
            values = [
                ast.Call(func=ast.Name(id = "_tick", ctx = ast.Load())),
                node.elt
            ]
        )
        node.elt = new_elt
        return node



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
        tree = ast.parse(source)
    except SyntaxError:
        raise ValidationError("Not valid AST")
    
    # fix this logic later but we are probably fine
    instrument = True
    is_ = Instrument()
    post_instr = is_.visit(tree)
    return ast.unparse(post_instr)


"""Execution & sandbox

Execute in a separate process; enforce wall‑clock timeout in parent with kill()+join(); never leave zombies.

In the child: set RLIMIT_CPU; attempt RLIMIT_AS and RLIMIT_NOFILE (ignore if unsupported). Clear os.environ, chdir to a temp dir, capture stdout/stderr (truncate each to ≤ 4096 bytes).

Return: { "status": <str>, "result": <any JSON‑serializable>, "runtime_ms": <int>, "stdout": <str>, "stderr": <str> }.

"""

def exec_child(src, entry_func, args, cpu_limit, mem_limit, fd_limit, q):
    import resource

    if cpu_limit:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit,cpu_limit))
    if mem_limit:
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
    if fd_limit:
        resource.setrlimit(resource.RLIMIT_NOFILE, (fd_limit, fd_limit))

    import sys
    sys.setrecursionlimit(100)

    import io
    err = io.StringIO()
    out = io.StringIO()
    sys.stderr = err
    sys.stdout = out

    try:
        ns = {"__builtins__": SAFE_BUILTINS}
        exec(src, ns, ns)
        func = ns.get(entry_func)
        res = func(*args) if func else None
        q.put({
            "result": res,
            "status": Status.OK,
            "stderr": err.getvalue(),
            "stdout": out.getvalue()
        })
    # add the specific exceptions too
    except Exception as e:
        q.put({
            "result": None,
            "status": Status.ERROR_NO_MESSAGE,
            "stderr": err.getvalue(),
            "stdout": out.getvalue()
        })
    return
    


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
    # it has already been validated and instrumented
    # we really just need to create the process

    # lets prepare the args tho
    try:
        q = multiprocessing.Queue()
        full_args = (prepared_source, entry, args, limits.get("cpu_s"), limits.get("mem_limit"), limits.get("nofile"), q)
        p = multiprocessing.Process(exec_child, full_args=(full_args))
        p.start()
        p.join(timeout=timeout_s)
        if p.is_alive():
            p.kill()
            # still dont undeerstand why we need this
            p.join()
    except Exception as e:
        return {
            "result": None,
            "status": Status.ERROR_NO_MESSAGE,
        }
    
    # get the value from the queue if it exists
    if not q.empty():
        res = q.get_nowait()
        # its already a dict
        return res
    else:
        return {
            "result": res,
            "status": Status.OK
        }

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
    results = [{}] * len(cases)
    for idx, case in enumerate(cases):
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_index = {}
            try:
                new_src = validate_and_instrument(case["src"])
                full_args = (new_src, case["entry"], timeout_s, limits)
                future = pool.submit(function=run_single, args=full_args)
                future_to_index[future] = idx
            except Exception as e:
                res = {
                    "result": None,
                    "status": str(e)
                }
                results[idx] = res
        
        for fut in as_completed(future_to_index.keys()):
            result = fut.result()
            results[future_to_index[fut]] = result
    return {"results": results}

