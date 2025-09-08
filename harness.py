import argparse, importlib, sys, json
from tests.adversarial_snippets import SNIPPETS

parser = argparse.ArgumentParser()
parser.add_argument("--impl", required=True, help="Path to your implementation directory (containing api.py)")
parser.add_argument("--mode", choices=["banlist","allowlist"], default="banlist")
parser.add_argument("--no-instrument", action="store_true")
parser.add_argument("--timeout", type=float, default=1.0)
args = parser.parse_args()

sys.path.insert(0, args.impl)
api = importlib.import_module("api")

SRC_OK = """
def square_sum(xs):
    return sum(x*x for x in xs)
"""

SRC_FUNC_NOT_FOUND = """
def other(): return 0
"""

SRC_LOOP = """
def loop_forever():
    while True:
        pass
"""

tests = [
    {"name":"sum squares", "src": SRC_OK, "func":"square_sum", "args":[[1,2,3]], "expected": 14},
    {"name":"missing func", "src": SRC_FUNC_NOT_FOUND, "func":"square_sum", "args":[[1,2,3]]},
    {"name":"infinite loop", "src": SRC_LOOP, "func":"loop_forever", "args":[]},
]

report = api.run_many( 
    tests,
    timeout_s=args.timeout,
    limits={"cpu_s":1, "as_bytes":256*1024*1024, "op_limit": 10000},
    mode=args.mode,
    op_limit=10000,
    instrument=not args.no_instrument
)
print(json.dumps(report, indent=2))

print("\\nAdversarial smoke:")
for name, code in SNIPPETS.items():
    res = api.run_many([{"name": name, "src": code, "func":"", "args":[]}],
                       timeout_s=args.timeout,
                       limits={"cpu_s":1, "as_bytes":128*1024*1024, "op_limit": 10000},
                       mode=args.mode,
                       op_limit=5000,
                       instrument=not args.no_instrument)
    print(name, res["results"][0]["result"]["status"])