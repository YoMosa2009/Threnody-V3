"""Tool schemas ("harnesses") for AgentLoop, so trajectories and evals cover many CLI-agent styles.

Each schema is a dict:
  name, system, tools (OpenAI function list, or None for text protocols), text_protocol (bool),
  dispatch(root, tool_name, args) -> (op, result)  where op is "edit", "run:<command>", "read" or "other".

Training schemas: agentloop, openhands, sweagent, minisweagent, claudecode, editblock (+ renamed variants).
Held-out (eval only, never in training data): opencode, codex.
"""
import fnmatch
import json
import os
import random
import re
import shlex
import subprocess

MAX_OUT = 6000


def clip(s):
    return s if len(s) <= MAX_OUT else s[:MAX_OUT // 2] + "\n...[truncated]...\n" + s[-MAX_OUT // 2:]


def inside(root, rel):
    p = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([os.path.realpath(root), p]) != os.path.realpath(root):
        raise PermissionError("path escapes the repository")
    return p


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": t} for k, t in props.items()}, "required": required}}}


# ---------------------------------------------------------------- core operations
def op_run(root, cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=timeout)
        return clip(f"exit={r.returncode}\n--- stdout\n{r.stdout}\n--- stderr\n{r.stderr}")
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout}s"


def op_read(root, path, offset=None, limit=None):
    lines = open(inside(root, path), encoding="utf-8").read().splitlines()
    start = max(int(offset or 1), 1)
    end = start - 1 + int(limit) if limit else len(lines)
    return clip("\n".join(f"{i:4d}| {l}" for i, l in enumerate(lines[start - 1:end], start)) or "(empty file)")


def op_write(root, path, content):
    p = inside(root, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8", newline="\n").write(content)
    return f"wrote {path} ({len(content)} chars)"


def op_replace(root, path, old, new, replace_all=False):
    p = inside(root, path)
    text = open(p, encoding="utf-8").read()
    n = text.count(old)
    if n == 0 or (n > 1 and not replace_all):
        return f"error: old string found {n} times (must be exactly 1{' or use replace_all' if n > 1 else ''})"
    open(p, "w", encoding="utf-8", newline="\n").write(text.replace(old, new) if replace_all else text.replace(old, new, 1))
    return f"edited {path}"


def op_insert(root, path, line, text):
    p = inside(root, path)
    lines = open(p, encoding="utf-8").read().split("\n")
    line = max(0, min(int(line), len(lines)))
    lines[line:line] = text.split("\n")
    open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    return f"inserted {len(text.splitlines())} line(s) after line {line} of {path}"


def op_edit_lines(root, path, start, end, text):
    p = inside(root, path)
    lines = open(p, encoding="utf-8").read().split("\n")
    s, e = int(start), int(end)
    if not (1 <= s <= e + 1 <= len(lines) + 1):
        return f"error: invalid line range {s}-{e} (file has {len(lines)} lines)"
    lines[s - 1:e] = text.split("\n") if text else []
    open(p, "w", encoding="utf-8", newline="\n").write("\n".join(lines))
    return f"replaced lines {s}-{e} of {path}"


def op_list(root, path="."):
    out = []
    base = inside(root, path)
    for d, dirs, files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x != "__pycache__")
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return clip("\n".join(out) or "(empty)")


def op_glob(root, pattern):
    files = op_list(root).splitlines()
    pat = pattern.replace("**/", "*")
    return "\n".join(f for f in files if fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(os.path.basename(f), pat)) or "(no matches)"


def op_grep(root, pattern, path=".", glob=None):
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"error: bad regex: {e}"
    hits = []
    for f in op_list(root, path).splitlines():
        if glob and not fnmatch.fnmatch(os.path.basename(f), glob):
            continue
        try:
            for i, l in enumerate(open(os.path.join(root, f), encoding="utf-8"), 1):
                if rx.search(l):
                    hits.append(f"{f}:{i}: {l.rstrip()}")
        except (UnicodeDecodeError, OSError):
            continue
    return clip("\n".join(hits) or "(no matches)")


def op_apply_patch(root, patch):
    """Codex-style patch: *** Begin Patch / *** Add File / *** Update File / *** Delete File / *** End Patch."""
    lines = patch.strip("\n").split("\n")
    if not lines or lines[0].strip() != "*** Begin Patch":
        return "error: patch must start with '*** Begin Patch'"
    i, done = 1, []
    while i < len(lines) and lines[i].strip() != "*** End Patch":
        h = lines[i]
        if h.startswith("*** Add File: "):
            path, body = h[14:].strip(), []
            i += 1
            while i < len(lines) and not lines[i].startswith("***"):
                body.append(lines[i][1:] if lines[i].startswith("+") else lines[i])
                i += 1
            op_write(root, path, "\n".join(body) + "\n")
            done.append(f"A {path}")
        elif h.startswith("*** Delete File: "):
            os.remove(inside(root, h[17:].strip()))
            done.append(f"D {h[17:].strip()}")
            i += 1
        elif h.startswith("*** Update File: "):
            path = h[17:].strip()
            p = inside(root, path)
            text = open(p, encoding="utf-8").read()
            i += 1
            hunks, cur = [], None
            while i < len(lines) and not lines[i].startswith("*** "):
                l = lines[i]
                if l.startswith("@@"):
                    cur = ([], [])
                    hunks.append(cur)
                else:
                    if cur is None:
                        cur = ([], [])
                        hunks.append(cur)
                    tag, body = (l[:1], l[1:]) if l[:1] in "+- " else (" ", l)
                    if tag in " -":
                        cur[0].append(body)
                    if tag in " +":
                        cur[1].append(body)
                i += 1
            for old, new in hunks:
                o, n = "\n".join(old), "\n".join(new)
                if text.count(o) != 1:
                    return f"error: hunk context not found exactly once in {path}"
                text = text.replace(o, n, 1)
            open(p, "w", encoding="utf-8", newline="\n").write(text)
            done.append(f"M {path}")
        else:
            return f"error: unexpected patch line: {h[:80]}"
    return "applied patch:\n" + "\n".join(done)


# ---------------------------------------------------------------- schemas
def _agentloop():
    tools = [
        _fn("bash", "Run a shell command in the repository root. Returns exit code, stdout and stderr (truncated).",
            {"command": "string"}, ["command"]),
        _fn("read_file", "Read a text file (relative path). Returns its content with line numbers.", {"path": "string"}, ["path"]),
        _fn("write_file", "Create or overwrite a file with the given content.", {"path": "string", "content": "string"},
            ["path", "content"]),
        _fn("edit_file", "Replace exactly one occurrence of old_str with new_str in a file.",
            {"path": "string", "old_str": "string", "new_str": "string"}, ["path", "old_str", "new_str"]),
    ]

    def dispatch(root, name, a):
        if name == "bash":
            return "run:" + a["command"], op_run(root, a["command"])
        if name == "read_file":
            return "read", op_read(root, a["path"])
        if name == "write_file":
            return "edit", op_write(root, a["path"], a["content"])
        if name == "edit_file":
            return "edit", op_replace(root, a["path"], a["old_str"], a["new_str"])
        return "other", f"error: unknown tool {name}"
    return tools, dispatch


def _openhands():
    tools = [
        _fn("execute_bash", "Execute a bash command in the terminal (cwd = repository root).", {"command": "string"}, ["command"]),
        {"type": "function", "function": {"name": "str_replace_editor", "description":
            "Custom editing tool for viewing, creating and editing files. Commands: view (file or directory), create "
            "(needs file_text), str_replace (old_str must match exactly once), insert (insert new_str after insert_line).",
            "parameters": {"type": "object", "properties": {
                "command": {"type": "string", "enum": ["view", "create", "str_replace", "insert"]},
                "path": {"type": "string"}, "file_text": {"type": "string"}, "old_str": {"type": "string"},
                "new_str": {"type": "string"}, "insert_line": {"type": "integer"},
                "view_range": {"type": "array", "items": {"type": "integer"}}},
                "required": ["command", "path"]}}},
        _fn("finish", "Signal that the task is complete, with a short summary.", {"message": "string"}, ["message"]),
    ]

    def dispatch(root, name, a):
        if name == "execute_bash":
            return "run:" + a["command"], op_run(root, a["command"])
        if name == "finish":
            return "finish", "ok"
        if name == "str_replace_editor":
            c, path = a.get("command"), a.get("path", "")
            if c == "view":
                if os.path.isdir(inside(root, path)):
                    return "read", op_list(root, path)
                vr = a.get("view_range") or [None, None]
                lim = (vr[1] - vr[0] + 1) if vr[0] and vr[1] and vr[1] > 0 else None
                return "read", op_read(root, path, vr[0], lim)
            if c == "create":
                return "edit", op_write(root, path, a.get("file_text", ""))
            if c == "str_replace":
                return "edit", op_replace(root, path, a.get("old_str", ""), a.get("new_str", ""))
            if c == "insert":
                return "edit", op_insert(root, path, a.get("insert_line", 0), a.get("new_str", ""))
            return "other", f"error: unknown command {c}"
        return "other", f"error: unknown tool {name}"
    return tools, dispatch


def _sweagent():
    tools = [
        _fn("bash", "Run a bash command.", {"command": "string"}, ["command"]),
        _fn("open", "Open a file and show up to 100 lines starting at line_number (default 1).",
            {"path": "string", "line_number": "integer"}, ["path"]),
        _fn("find_file", "Find files whose name matches file_name (glob) in the repository.", {"file_name": "string"}, ["file_name"]),
        _fn("search_dir", "Search for a regex in all files under dir (default '.').", {"search_term": "string", "dir": "string"},
            ["search_term"]),
        _fn("edit", "Replace lines start_line..end_line (inclusive, 1-based) of path with replacement text.",
            {"path": "string", "start_line": "integer", "end_line": "integer", "replacement": "string"},
            ["path", "start_line", "end_line", "replacement"]),
        _fn("create", "Create a new file with content.", {"path": "string", "content": "string"}, ["path", "content"]),
        _fn("submit", "Submit your changes when the task is done.", {}, []),
    ]

    def dispatch(root, name, a):
        if name == "bash":
            return "run:" + a["command"], op_run(root, a["command"])
        if name == "open":
            return "read", op_read(root, a["path"], a.get("line_number") or 1, 100)
        if name == "find_file":
            return "read", op_glob(root, a["file_name"])
        if name == "search_dir":
            return "read", op_grep(root, a["search_term"], a.get("dir") or ".")
        if name == "edit":
            return "edit", op_edit_lines(root, a["path"], a["start_line"], a["end_line"], a["replacement"])
        if name == "create":
            return "edit", op_write(root, a["path"], a["content"])
        if name == "submit":
            return "finish", "submitted"
        return "other", f"error: unknown tool {name}"
    return tools, dispatch


def _minisweagent():
    tools = [_fn("bash", "Execute a bash command. Every action must be a single shell command; use heredocs or sed to edit "
                 "files.", {"command": "string"}, ["command"])]

    def dispatch(root, name, a):
        if name == "bash":
            return "run:" + a["command"], op_run(root, a["command"])
        return "other", f"error: unknown tool {name}"
    return tools, dispatch


def _claudecode():
    tools = [
        _fn("Read", "Read a file from the local filesystem. Returns lines with line numbers.",
            {"file_path": "string", "offset": "integer", "limit": "integer"}, ["file_path"]),
        _fn("Write", "Write a file to the local filesystem (overwrites).", {"file_path": "string", "content": "string"},
            ["file_path", "content"]),
        _fn("Edit", "Exact string replacement in a file. old_string must be unique unless replace_all is true.",
            {"file_path": "string", "old_string": "string", "new_string": "string", "replace_all": "boolean"},
            ["file_path", "old_string", "new_string"]),
        _fn("Bash", "Execute a bash command in the project directory.", {"command": "string", "description": "string"},
            ["command"]),
        _fn("Grep", "Search file contents with a regular expression.", {"pattern": "string", "path": "string", "glob": "string"},
            ["pattern"]),
        _fn("Glob", "Find files by glob pattern.", {"pattern": "string"}, ["pattern"]),
    ]

    def dispatch(root, name, a):
        if name == "Read":
            return "read", op_read(root, a["file_path"], a.get("offset"), a.get("limit"))
        if name == "Write":
            return "edit", op_write(root, a["file_path"], a["content"])
        if name == "Edit":
            return "edit", op_replace(root, a["file_path"], a["old_string"], a["new_string"], bool(a.get("replace_all")))
        if name == "Bash":
            return "run:" + a["command"], op_run(root, a["command"])
        if name == "Grep":
            return "read", op_grep(root, a["pattern"], a.get("path") or ".", a.get("glob"))
        if name == "Glob":
            return "read", op_glob(root, a["pattern"])
        return "other", f"error: unknown tool {name}"
    return tools, dispatch


def _opencode():
    tools = [
        _fn("read", "Read a file. Returns numbered lines.", {"filePath": "string", "offset": "integer", "limit": "integer"},
            ["filePath"]),
        _fn("write", "Write content to a file.", {"filePath": "string", "content": "string"}, ["filePath", "content"]),
        _fn("edit", "Replace oldString with newString in a file.",
            {"filePath": "string", "oldString": "string", "newString": "string", "replaceAll": "boolean"},
            ["filePath", "oldString", "newString"]),
        _fn("bash", "Run a shell command.", {"command": "string", "timeout": "integer"}, ["command"]),
        _fn("grep", "Regex search across files.", {"pattern": "string", "include": "string"}, ["pattern"]),
        _fn("list", "List files in a directory.", {"path": "string"}, []),
    ]

    def dispatch(root, name, a):
        if name == "read":
            return "read", op_read(root, a["filePath"], a.get("offset"), a.get("limit"))
        if name == "write":
            return "edit", op_write(root, a["filePath"], a["content"])
        if name == "edit":
            return "edit", op_replace(root, a["filePath"], a["oldString"], a["newString"], bool(a.get("replaceAll")))
        if name == "bash":
            return "run:" + a["command"], op_run(root, a["command"])
        if name == "grep":
            return "read", op_grep(root, a["pattern"], ".", a.get("include"))
        if name == "list":
            return "read", op_list(root, a.get("path") or ".")
        return "other", f"error: unknown tool {name}"
    return tools, dispatch


def _codex():
    tools = [
        {"type": "function", "function": {"name": "shell", "description": "Run a command. `command` is an argv array, "
         "e.g. [\"bash\", \"-lc\", \"pytest -q\"].", "parameters": {"type": "object", "properties": {
             "command": {"type": "array", "items": {"type": "string"}}}, "required": ["command"]}}},
        _fn("apply_patch", "Apply a patch in this format:\n*** Begin Patch\n*** Update File: path\n@@\n context\n-old line\n"
            "+new line\n*** Add File: path\n+content\n*** End Patch", {"input": "string"}, ["input"]),
    ]

    def dispatch(root, name, a):
        if name == "shell":
            argv = a["command"] if isinstance(a["command"], list) else [str(a["command"])]
            cmd = argv[-1] if len(argv) >= 3 and argv[0] in ("bash", "sh") and argv[1] in ("-lc", "-c") else \
                shlex.join(argv)
            return "run:" + cmd, op_run(root, cmd)
        if name == "apply_patch":
            return "edit", op_apply_patch(root, a["input"])
        return "other", f"error: unknown tool {name}"
    return tools, dispatch


EDITBLOCK_SYSTEM = (
    "You are an expert software engineer editing a repository. You cannot call tools. To change files, reply with "
    "SEARCH/REPLACE blocks; to run commands, reply with ```bash blocks (one command per block). Format:\n\n"
    "path/to/file.py\n```python\n<<<<<<< SEARCH\nexact existing lines\n=======\nnew lines\n>>>>>>> REPLACE\n```\n\n"
    "An empty SEARCH section creates a new file. You will receive the results of your edits and commands in the next "
    "message. Verify your work by running tests. When the task is done, reply with a short summary and no blocks.")
BLOCK_RE = re.compile(r"^([^\n`]+?)\n```[^\n]*\n<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE\n```", re.S | re.M)
BASH_RE = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.S)


def editblock_step(root, text):
    """Apply SEARCH/REPLACE blocks and run bash blocks from a text reply. Returns (ops, feedback) or (None, None)."""
    ops, fb = [], []
    for path, search, replace in BLOCK_RE.findall(text):
        path = path.strip().strip("`*")
        try:
            if not search.strip():
                fb.append(op_write(root, path, replace))
            else:
                fb.append(op_replace(root, path, search, replace))
            ops.append(("edit", path))
        except Exception as e:
            fb.append(f"error editing {path}: {type(e).__name__}: {e}")
    for cmd in BASH_RE.findall(text):
        cmd = cmd.strip()
        fb.append(f"$ {cmd}\n{op_run(root, cmd)}")
        ops.append(("run:" + cmd, ""))
    if not ops:
        return None, None
    return ops, "\n\n".join(fb)


SYSTEMS = {
    "agentloop": ("You are a coding agent working in a repository on the user's machine. The current working directory is "
                  "the repository root. Use the tools to inspect files, make changes and run commands (python and pytest "
                  "are available). Verify your work by running code or tests. When the task is complete, reply with a "
                  "short summary and no tool calls."),
    "openhands": ("You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks. The "
                  "repository is in the current working directory. Explore, fix, and verify with tests; call finish when done."),
    "sweagent": ("You are an autonomous programmer working directly in a command line with a special interface to open, "
                 "search and edit files. Solve the issue, run the tests to confirm, then call submit."),
    "minisweagent": ("You are a helpful assistant that can interact with a computer. You can only run bash commands, one per "
                     "action. Inspect the repository, make the fix, verify it by running tests, and when finished reply "
                     "without a tool call."),
    "claudecode": ("You are an interactive CLI tool that helps users with software engineering tasks. Use the available "
                   "tools to read, search and edit code and run commands. Prefer Edit over Write for existing files. Run the "
                   "tests before you say the task is done. Keep your final answer brief."),
    "opencode": ("You are opencode, an interactive CLI coding agent. Use the tools to explore and modify the project in the "
                 "current directory. Always validate changes by running the project's tests. Be concise."),
    "codex": ("You are a coding agent running in a terminal-based environment. Use `shell` to run commands and `apply_patch` "
              "to edit files. Verify changes with the test suite before finishing; then summarize briefly."),
    "editblock": EDITBLOCK_SYSTEM,
}
_BUILDERS = {"agentloop": _agentloop, "openhands": _openhands, "sweagent": _sweagent, "minisweagent": _minisweagent,
             "claudecode": _claudecode, "opencode": _opencode, "codex": _codex}
TRAIN_SCHEMAS = ["agentloop", "openhands", "sweagent", "minisweagent", "claudecode", "editblock"]
HELDOUT_SCHEMAS = ["opencode", "codex"]


def get(name):
    if name == "editblock":
        return {"name": name, "system": EDITBLOCK_SYSTEM, "tools": None, "text_protocol": True, "dispatch": None}
    tools, dispatch = _BUILDERS[name]()
    return {"name": name, "system": SYSTEMS[name], "tools": tools, "text_protocol": False, "dispatch": dispatch}


# ---------------------------------------------------------------- renamed variants (read the schema, don't memorize it)
_TOOL_SYNS = ["run", "exec", "sh", "terminal", "cmd", "view", "cat_file", "open_file", "show", "put_file", "save_file",
              "patch_file", "modify", "replace_text", "search", "find", "ls", "tree", "lookup", "files", "done", "complete"]
_ARG_SYNS = ["cmd", "script", "target", "file", "filename", "fp", "body", "text", "data", "before", "after", "find",
             "replace_with", "query", "pattern_str", "dir_path", "start", "stop", "lineno", "payload"]


def renamed(schema, seed):
    """Same behaviour, randomized tool/argument names (and shuffled tool order)."""
    if schema["text_protocol"]:
        return schema
    rng = random.Random(seed)
    tnames = rng.sample(_TOOL_SYNS, len(schema["tools"]))
    tmap, amaps, tools = {}, {}, []
    for t, new in zip(schema["tools"], tnames):
        f = t["function"]
        props = f["parameters"]["properties"]
        new_args = rng.sample(_ARG_SYNS, len(props)) if props else []
        amap = dict(zip(new_args, props))  # new -> old
        tmap[new], amaps[new] = f["name"], amap
        inv = {v: k for k, v in amap.items()}
        tools.append({"type": "function", "function": {
            "name": new, "description": f["description"],
            "parameters": {"type": "object", "properties": {inv[k]: v for k, v in props.items()},
                           "required": [inv[k] for k in f["parameters"].get("required", [])]}}})
    rng.shuffle(tools)
    inner = schema["dispatch"]

    def dispatch(root, name, a):
        if name not in tmap:
            return "other", f"error: unknown tool {name}"
        return inner(root, tmap[name], {amaps[name].get(k, k): v for k, v in a.items()})
    return dict(schema, name=schema["name"] + "~r", tools=tools, dispatch=dispatch)


# ---------------------------------------------------------------- trajectory checks
VERIFY_RE = re.compile(r"\b(pytest|python3?|unittest)\b")
SHELL_EDIT_RE = re.compile(r"(>|sed -i|tee |cat <<|perl -pi)")


def verified(ops):
    """True if a python/pytest run happened after the last edit (ops: list of op strings in order)."""
    last_edit = last_run = -1
    for i, op in enumerate(ops):
        if op == "edit":
            last_edit = i
        elif op.startswith("run:"):
            cmd = op[4:]
            if SHELL_EDIT_RE.search(cmd):
                last_edit = i
            if VERIFY_RE.search(cmd):
                last_run = i
    return last_run > last_edit >= 0 or (last_edit < 0 and last_run >= 0)


def max_repeat(messages):
    """Longest run of consecutive identical assistant actions (tool calls or text replies)."""
    best = run = 0
    prev = None
    for m in messages:
        if m.get("role") != "assistant":
            continue
        calls = m.get("tool_calls") or []
        key = json.dumps([(c["function"]["name"], c["function"].get("arguments")) for c in calls], sort_keys=True) \
            if calls else (m.get("content") or "").strip()[:2000]
        run = run + 1 if key == prev and key else 1
        prev = key
        best = max(best, run)
    return best
