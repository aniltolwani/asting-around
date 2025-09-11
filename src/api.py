import ast
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor
import time


SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "range": range, "len": len, "print": print,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "all": all, "any": any, "isinstance": isinstance,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "Exception": Exception
}

BANLIST = (
    "eval",
    "exec",
    "open",
    "__builtins__",
    # any dunder operation
    # we need to ban imports do but we will do that in the validator
)
import multiprocessing
# use fork since the harness is messed up 
multiprocessing.set_start_method("fork", force = True)
GLOBAL_MAX = 10_000
CNT = 0

class BanListError(Exception):
    pass

class AllowListError(Exception):
    pass

class BanListValidator(ast.NodeVisitor):
    def visit_Import(self, node):
        raise BanListError("No imports allowed")

    def visit_ImportFrom(self, node):
        raise BanListError("No imports from allowed")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in BANLIST:
                raise BanListError(f"{node.func.id} is not allowed")
            elif "__" in node.func.id:
                raise BanListError("No dunder methods allowed")
        return self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            raise BanListError("No dunder attributes allowed")
        return self.generic_visit(node)

class AllowListValidator(ast.NodeVisitor):
    def visit_Import(self, node):
        raise AllowListError("No imports allowed")

    def visit_ImportFrom(self, node):
        raise AllowListError("No imports from allowed")

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_BUILTINS:
                raise AllowListError("No non builtin methods allowed")
        return self.generic_visit(node)
    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            raise AllowListError("No dunder attributes allowed")
        return self.generic_visit(node)

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

    def visit_For(self, node):
        new_node = self.generic_visit(node)
        expr = ast.Expr(
            value = ast.Call(
                func = ast.Name(id = "_tick_def", ctx = ast.Load()),
                args = [],
                keywords=[]
            )
        )
        new_node.body = [expr] + new_node.body
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
        new_node.body = [expr] + new_node.body
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
        new_node.body = [expr] + new_node.body
        return new_node

def validate_and_instrument(src: str, mode: str = "banlist", op_limit: int = 10000, instrument: bool = True) -> str:
    print(f"Validating and instrumenting code in {mode} mode with op_limit={op_limit}")
    
    # first, just parse the code
    try:
        parsed = ast.parse(src)
        print("Code parsed successfully")
    except SyntaxError as e:
        print(f"Syntax error during parsing: {e}")
        raise SyntaxError("The code you submitted is not parseable.")

    # validate
    print(f"Starting validation with mode: {mode}")
    if mode == "banlist":
        banlist = BanListValidator()
        banlist.visit(parsed)
        print("Banlist validation passed")
    elif mode == "allowlist":
        allowlist = AllowListValidator()
        allowlist.visit(parsed)
        print("Allowlist validation passed")
    else:
        print(f"Invalid mode: {mode}")
        raise ValueError(f"We must have a mode to validate the code. {mode} is invalid.")

    # now, we need to instrument it 
    if instrument:
        print("Instrumenting code with loop counters")
        instrumenter = InstrumentLoops()
        new_tree = _build_and_repair(instrumenter.visit(parsed))
        instrumented_code = ast.unparse(new_tree)
        print(f"Code instrumented successfully, length: {len(instrumented_code)} chars")
        return instrumented_code
    else:
        print("Skipping instrumentation")
        unparsed_code = ast.unparse(parsed)
        print(f"Code unparsed without instrumentation, length: {len(unparsed_code)} chars")
        return unparsed_code

def child_main(src, entry_func, cpu_limit, memlimit, q, args):
    import resource
    import sys
    print(f"Child process starting: func={entry_func}, cpu_limit={cpu_limit}, args={args}")
    if cpu_limit:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        print(f"CPU limit set to {cpu_limit} seconds")
    # if memlimit:
        # not supported on Darwin
        # resource.setrlimit(resource.RLIMIT_AS, (memlimit,memlimit))
        
    sys.setrecursionlimit(100)
    print("Recursion limit set to 100")

    globals = {
        "__builtins__": SAFE_BUILTINS
    }

    try:
        print("Executing user code...")
        # string is fine
        exec(src, globals, globals)
        # get the function and execute it 
        func = globals.get(entry_func)
        if func:
            print(f"Function {entry_func} found, calling with args: {args}")
            result = func(*args)
            print(f"Function returned: {result}")
        else:
            print(f"Function {entry_func} not found in globals")
            result = None
        q.put({
            "result": result,
            "status": "OK"
        })
        print("Result put in queue successfully")
    except MemoryError:
        q.put({
            "result": None,
            "status": "memory_limit_exceeded"
        })
    except RecursionError:
        q.put({
            "result": None,
            "status": "Max recursion depth."
        })
    except Exception as e:
        if "ops exceeded" in str(e):
            q.put({
                "result": None,
                "status": "OP_EXCEEDED",
            })
        else:
            q.put({
                "result": None,
                "status": str(e),
            })
    return

# no one will call run one so we will
def run_one(src: str, entry_func: str, args: tuple, timeout_s=1.0, limits=None, op_limit=10_000) -> dict:
    print(f"Running single test: func={entry_func}, timeout={timeout_s}s, op_limit={op_limit}")

    # add a function call to entry_func

    if not limits:
        limits = {"cpu_s": 1, "as_bytes": 256 * 1024 * 1024}
        print("Using default limits")
    else:
        print(f"Using custom limits: {limits}")

    # just assume there is an op_limit for now, and give it access to a def

    # process
    # insert tick def as src code since its hard to pass a function, not pickable

    _tick_src = f"""
cnt = 0
op_limit = {op_limit}
def _tick_def():
    global cnt
    cnt += 1
    if cnt > op_limit:
        raise ValueError("ops exceeded")

"""

    src = _tick_src + src
    print("src we are gonna run", src)

    # new process
    print("Creating child process...")
    q = Queue() 
    p = Process(target = child_main, args = (src, entry_func, limits["cpu_s"], limits["as_bytes"], q, args))
    p.start()
    print(f"Child process started, waiting up to {timeout_s}s...")
    p.join(timeout=timeout_s)
    if p.is_alive():
        print("Process timed out, killing...")
        p.kill()
        return {
            "result": None,
            "status": "TIMEOUT",
        }
    else:
        result = q.get()
        return result

def run_many(testcases: list[dict], timeout_s=1.0, limits=None, mode="banlist", op_limit=10000, instrument=True) -> dict:
    print(f"Running {len(testcases)} test cases with mode={mode}, timeout={timeout_s}s")
    if op_limit or instrument:
        do_instrumentation = True
    else:
        do_instrumentation = False
    
    if limits and "op_limit" in limits:
        op_limit_real = max(op_limit, limits["op_limit"])
    else:
        op_limit_real = op_limit
    results = []
    for i, case in enumerate(testcases):
        try:
            instrumented = validate_and_instrument(case["src"], mode, op_limit, do_instrumentation)
            res = run_one(
                    instrumented, 
                    case.get("func", ""), 
                    case.get("args", []), 
                    timeout_s, 
                    limits, 
                    op_limit_real
                )
            results.append({"result": res})
        except Exception as e:
            res = {
                "result":{
                    "status": str(e),
                    "result": None,
                }
            }
            results.append(res)
    return {"results": results}
