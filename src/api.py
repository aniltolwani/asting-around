import ast
from multiprocessing import Process, Queue
import time


SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "range": range, "len": len, "print": print,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "all": all, "any": any, "isinstance": isinstance,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple
}

BANLIST = (
    "eval",
    "exec",
    "open",
    "__builtins__",
    # any dunder operation
    # we need to ban imports do but we will do that in the validator
)
GLOBAL_MAX = 10_000
CNT = 0

class BanListError(Exception):
    pass

class AllowListError(Exception):
    pass

class BanListValidator(ast.NodeVisitor):
    def visit_Import(self, node):
        # lets not let any imports in? 
        raise BanListError("No Imports allowed")

    def visit_ImportFrom(self, node):
        raise BanListError("No imports from allowed")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in BANLIST:
                raise BanListError(f"{node.func.id} is not allowed")
            elif "__" in node.func.id:
                raise BanListError("No dunder methods allowed")
        return self.generic_visit(node)

class AllowListValidator(ast.NodeVisitor):
    pass

def _build_and_repair(tree: ast.Module):
    ast.fix_missing_locations(tree)
    tree.type_ignores = []
    return tree

class InstrumentLoops(ast.NodeTransformer):
    """
    class for instrumenting:
        - functiondef
        - while
        - for
    """
    def __init__():
        pass

    def visit_For(self, node):
        new_node = self.generic_visit(node)
        expr = ast.Expr(
            value = ast.Call(
                func = ast.Name(id = "_tick_def", ctx = ast.Load()),
                args = [],
                keywords=[]
            )
        )
        new_body = [expr] + new_node.body
        new_node = ast.Module(body = new_body)
        return new_node

    def visit_FunctionDef(self, node):
        new_node = self.generic_visit(node)
        expr = ast.Expr(
            value = ast.Call(
                func = ast.Name(id = "_tick_def", ctx = ast.Load()),
                args = [],
                keywords=[]
            )
        )
        new_body = [expr] + new_node.body
        new_node = ast.Module(body = new_body)
        return new_node
    
    def visit_While(self, node):
        new_node = self.generic_visit(node)
        expr = ast.Expr(
            value = ast.Call(
                func = ast.Name(id = "_tick_def", ctx = ast.Load()),
                args = [],
                keywords=[]
            )
        )
        new_body = [expr] + new_node.body
        new_node = ast.Module(body = new_body)
        return new_node

def validate_and_instrument(src: str, mode: str = "banlist", op_limit: int = 10000, instrument: bool = True) -> str:
    
    # first, just parse the code
    try:
        parsed = ast.parse(src)
    except SyntaxError as e:
        raise SyntaxError("The code you submitted is not parseable.")

    # validate
    if mode == "banlist":
        banlist = BanListValidator()
        banlist.visit(parsed)
    elif mode == "allowlist":
        allowlist = AllowListValidator()
        allowlist.visit(parsed)
    else:
        raise ValueError(f"We must have a mode to validate the code. {mode} is invalid.")

    # now, we need to instrument it 
    instrument = InstrumentLoops()
    new_tree = _build_and_repair(instrument.visit(parsed))
    return ast.unparse(new_tree)

def _exec_child(src, ns, timeout, limits):
    ## need to pass this as src since ast trees are not pickle-able
    ## just runs executed code in a child container
    ## ns will be both locals and global

    p = Process(target=src)
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
    pass

# no one will call run one so we will
def run_one(src: str, entry_func: str, args: tuple, timeout_s=1.0, limits=None, op_limit=10_000) -> dict:

    # add a function call to entry_func
    parsed = ast.parse(src)

    final = ast.Assign(
        targets = [
            ast.Name(id="result", ctx = ast.Store())
        ],
        value = ast.Call(
            func = ast.Name(id = entry_func, ctx = ast.Load()),
            args = [ast.Constant(value = arg) for arg in args],
            keywords = [],
        )
    )

    # we need to execute both together
    new_body = parsed.body + [final]
    new_tree = _build_and_repair(ast.Module(new_body))
    # just assume there is an op_limit for now, and give it access to a def
    cnt = 0
    def _tick_def():
        nonlocal cnt
        cnt += 1
        if cnt > op_limit:
            raise ValueError
    # process
    # is this a thread? we probably want to spawn not fork
    # i think threading uses forks and multprocesses uses spawn?
    # so spawn would be from fresh, lets use that
    # process. join = timeout
    state = {
        "_tick_def": _tick_def,
        "__builtins__": SAFE_BUILTINS,
    }
    
    res = _exec_child(ast.unparse(new_tree), state, timeout_s, limits)
    # return back the function state with the result
    # what do we return? we dont know what variable the code is writing too.
    return {
        "result": state["result"],
        "status": 200,
    }

def run_many(testcases: list[dict], timeout_s=1.0, limits=None, mode="banlist", op_limit=10000, instrument=True) -> dict:
    if op_limit or instrument:
        do_instrumentation = True
    else:
        do_instrumentation = False
    
    op_limit_real = max(op_limit, limits.get(op_limit, 0))
    results = []
    for case in testcases:
        try:
            instrumented = validate_and_instrument(case["code"], mode, op_limit, do_instrumentation)
            results.append(run_one(instrumented, case["func"], case["args"], timeout_s, limits, op_limit_real))
        except Exception as e:
            res = {
                "status": e,
                "result": None,
            }
            results.append(res)
    return {"results": results}
