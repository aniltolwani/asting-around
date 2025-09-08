import ast
from multiprocessing import Value

ALLOWLIST = (
    # ternary addition / subtraction operators
    # variable store and load and writes
    # function defs, basic operations
    # string, ints, etc.
)

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
)
GLOBAL_MAX = 10_000
CNT = 0

class BanListError(Exception):
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

        return self.generic_visit

    def 




class OperationCounter(ast.NodeTransformer):
    def __init__(self):
        self.cnt = 0

    def wrap_tick(self, node):
        # wraps node with a tick wrapper
        return ast.Subscript(
            value = ast.Tuple(
                elts = [
                    ast.Call(func = ast.Name("_tick", ctx = ast.Load()), args = [], keywords=[]),
                    node
                ],
                ctx = ast.Load()
            ),
            slice = ast.Constant(value = 1),
            ctx = ast.Load()
        )
        
    def visit_BinOp(self, node):
        new_node =  self.generic_visit(node)
        # since this is a subclass of nodeTransformer (write)
        # we must return a node, ideally the one modified by generic visit
        
        return self.wrap_tick(new_node)



def validate_and_instrument(src: str, mode: str = "banlist", op_limit: int = 10000, instrument: bool = True) -> str:
    
    # first, just parse the code

    parsed = ast.parse(src)

    # now, we need to instrument it 

    # to do this, we can use the tuple pattern: (_tick, func)[1] will let us execute tick + run and return the same function

    # nodetransformer also lets us instrument any op: ie visit_BinOp will be run on all +/*/- operation
    
    pass


def run_one(src: str, entry_func: str, args: tuple, timeout_s=1.0, limits=None) -> dict:

    # parse the src code
    try:
        parsed_tree = ast.parse(src)
    except SyntaxError as e:
        return {"result": None, "status": f"syntax error: {e}"}

    # add a function call to entry_func

    final = ast.Assign(
        targets = [
            ast.Name(id="result", ctx = ast.Store())
        ],
        value = ast.Call(
            # this needs to be a named reference to the variable 
            # that is loaded from the string contained in entry func
            func = ast.Name(id = entry_func, ctx = ast.Load()),
            # this needs to be a list ie ast.Constant
            # but we don't know if it is a constant a priori...
            # hmmm
            args = [ast.Constant(value = arg) for arg in args],
            keywords = [],
        )
    )

    # we need to execute both together
    # parsed_tree is a generic module so we need to use a new module
    new_body = parsed_tree.body + [final]
    new_tree = ast.Module(new_body)

    # instrument is for cpu_s, as_bytes, and op_count
    state = {}

    if limits:
        # limits is a dict with cpu_s, as_bytes, and op_limit
        op_lim = limits.get("op_limit")
        if op_lim:
            CNT = 0
            def _tick():
                nonlocal CNT
                CNT += 1
                if CNT >= op_lim:
                    raise ValueError
            counter = OperationCounter()
            new_tree = counter.visit(new_tree)
            state["_tick"] = _tick


    ast.fix_missing_locations(new_tree)
    new_tree.type_ignores = []
    # build a state with everything we need

    exec(compile(new_tree, "<ast>", "exec"), state)
    # return back the function state with the result
    # what do we return? we dont know what variable the code is writing too.
    return {
        "result": state["result"],
        "status": 200,
    }

def run_many(testcases: list[dict], timeout_s=1.0, limits=None, mode="banlist", op_limit=10000, instrument=True) -> dict:
    pass
