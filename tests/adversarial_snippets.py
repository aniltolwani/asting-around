# Adversarial snippets for acceptance
SNIPPETS = {
    "dunder_attr": "(1).__class__",
    "dyn_import": "__import__('os').system('echo hacked')",
    "file_open": "open('x.txt','w').write('hi')",
    "globals_eval": "globals()['__builtins__']['eval']('1+1')",
    "recursion": "def f(n): return f(n+1)\nf(0)",
    "mem_bomb": "x=[0]* (50_000_000)",
    "while_true": "def spin():\n    while True: pass\nspin()"
}