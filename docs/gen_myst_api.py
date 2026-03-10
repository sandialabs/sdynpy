#!/usr/bin/env python3
"""
myst_api_gen.py

MyST Markdown API documentation generator for Jupyter Book v2.

This version adds:
- repo layout support for code under: /src/<package_name>
- "public alias" (re-export) handling from:
    * package __init__.py
    * any subpackage __init__.py
  so docs prefer canonical paths users import from (e.g., pkg.Foo) even if defined in pkg.mod.Foo.
- Still documents everything discovered in the package modules (not only aliased objects).
- Function pages for module-level functions only; class methods stay on the class page.

How aliasing works (practical behavior):
- We walk every package module and submodule to discover real objects (definition inventory).
- We also scan each package/subpackage __init__.py module for re-exports:
    - names in __all__ if present, else public names not starting with "_"
    - if that name refers to an object whose __module__ starts with your package, we record an alias.
- When generating a page for an object, we compute a "preferred name":
    - if the object is re-exported anywhere, we choose the *shortest* alias path (fewest segments),
      tie-breaker: prefer the top-level package alias (pkg.X) over deeper ones.
- Pages are written using preferred names for titles, targets, and cross-links.
- We also include an "Also available as:" list when multiple aliases exist.

Caveat:
- Python can create multiple distinct objects that look similar (wrappers, functools.partial,
  C-extension objects without source). We only alias objects where identity matches and module
  belongs to your package.

Run:
  python gen_myst_api.py --package yourpkg --out docs/api \
    --repo https://github.com/ORG/REPO --repo-root /path/to/repo --ref main

For /src layout, set --repo-root to the repository root. GitHub links will match src paths.

If you want help integrating into your Jupyter Book _toc.yml, see:
https://wp.sandia.gov/sandia-ai-chat/how-to-use/
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import pkgutil
import re
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ----------------------------
# NumPyDoc parsing (stdlib-only)
# ----------------------------

NUMPY_SECTION_UNDERLINE = re.compile(r"^[-=]{3,}\s*$")


@dataclasses.dataclass
class NumpyParam:
    name: str
    type: str
    desc: str


@dataclasses.dataclass
class NumpySection:
    title: str
    body: str = ""
    params: List[NumpyParam] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class NumpyDoc:
    summary: str = ""
    extended_summary: str = ""
    sections: Dict[str, NumpySection] = dataclasses.field(default_factory=dict)


def _split_numpydoc_sections(doc: str) -> List[Tuple[str, List[str]]]:
    lines = doc.splitlines()
    out: List[Tuple[str, List[str]]] = []
    cur_title = ""
    cur: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() and (i + 1) < len(lines) and NUMPY_SECTION_UNDERLINE.match(lines[i + 1]):
            out.append((cur_title, cur))
            cur_title = line.strip()
            cur = []
            i += 2
            if i < len(lines) and not lines[i].strip():
                i += 1
            continue
        cur.append(line)
        i += 1

    out.append((cur_title, cur))
    return out


def _parse_param_block(lines: List[str]) -> List[NumpyParam]:
    params: List[NumpyParam] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        m = re.match(r"^(\S.*?)(\s*:\s*)(.*\S)?\s*$", line)
        if m and (not line.startswith(" ")):
            name = m.group(1).strip()
            typ = (m.group(3) or "").strip()
            i += 1
            desc_lines: List[str] = []
            while i < len(lines):
                if not lines[i].strip():
                    desc_lines.append("")
                    i += 1
                    continue
                if lines[i].startswith(" " * 4) or lines[i].startswith("\t"):
                    desc_lines.append(lines[i].strip())
                    i += 1
                else:
                    break
            desc = "\n".join(desc_lines).strip()
            params.append(NumpyParam(name=name, type=typ, desc=desc))
        else:
            params.append(NumpyParam(name=line.strip(), type="", desc=""))
            i += 1
    return params


def parse_numpydoc(doc: str) -> NumpyDoc:
    doc = (doc or "").strip("\n")
    nd = NumpyDoc()
    if not doc.strip():
        return nd

    parts = _split_numpydoc_sections(doc)
    lead_title, lead_lines = parts[0]
    lead_text = "\n".join(lead_lines).strip("\n")
    if lead_text:
        lead_paras = [p.strip() for p in re.split(r"\n\s*\n", lead_text) if p.strip()]
        if lead_paras:
            nd.summary = lead_paras[0].replace("\n", " ").strip()
            if len(lead_paras) > 1:
                nd.extended_summary = "\n\n".join(lead_paras[1:]).strip()

    for title, sec_lines in parts[1:]:
        body = "\n".join(sec_lines).rstrip()
        sec = NumpySection(title=title, body=body)

        if title in (
            "Parameters",
            "Other Parameters",
            "Returns",
            "Yields",
            "Raises",
            "Warns",
            "Attributes",
            "Methods",
            "See Also",
        ):
            sec.params = _parse_param_block(sec_lines)

        nd.sections[title] = sec

    return nd


# ----------------------------
# Utilities
# ----------------------------

def is_private_name(name: str) -> bool:
    return name.startswith("_") and name not in ("__init__",)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def slugify(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "-", name)

def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default

def parse_all(module: types.ModuleType) -> Optional[List[str]]:
    val = safe_getattr(module, "__all__", None)
    if isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
        return list(val)
    return None

def get_doc(obj: Any) -> str:
    return inspect.getdoc(obj) or ""

def first_sentence_or_line(doc: str) -> str:
    if not doc:
        return ""
    nd = parse_numpydoc(doc)
    if nd.summary:
        return nd.summary
    for line in doc.strip().splitlines():
        s = line.strip()
        if s:
            return s
    return ""

def format_signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "()"

def qualname(obj: Any) -> str:
    qn = safe_getattr(obj, "__qualname__", None)
    if isinstance(qn, str):
        return qn
    return safe_getattr(obj, "__name__", obj.__class__.__name__)

def obj_module_name(obj: Any) -> str:
    return safe_getattr(obj, "__module__", "") or ""

def indent_md(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else "" for line in text.splitlines())


# ----------------------------
# GitHub source linking (/src layout supported implicitly)
# ----------------------------

@dataclasses.dataclass
class SourceRef:
    file_abs: Path
    file_rel_repo: str
    start_line: int
    end_line: int

def relpath_posix(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()

def get_source_ref(obj: Any, repo_root: Optional[Path]) -> Optional[SourceRef]:
    if repo_root is None:
        return None
    try:
        file_path = inspect.getsourcefile(obj) or inspect.getfile(obj)
        if not file_path:
            print(f"Could not find the file path for {obj=}")
            print(f"With {repo_root=}")
            return None
        file_abs = Path(file_path).resolve()
        if not file_abs.exists():
            print(f"File path {file_abs=} does not exist for {obj=}")
            print(f"With {repo_root=}")
            return None

        try:
            src_lines, start = inspect.getsourcelines(obj)
            end = start + len(src_lines) - 1
        except (OSError, TypeError):
            print(f"Could not get source lines for {obj=}")
            print(f"With {repo_root=}")
            start, end = 1, 1

        rr = repo_root.resolve()
        try:
            file_rel_repo = relpath_posix(file_abs, rr)
        except Exception as e:
            print(f"Could not find relative path for {obj=}")
            print(f"With {repo_root=}")
            print(e)
            return None

        return SourceRef(file_abs=file_abs, file_rel_repo=file_rel_repo, start_line=start, end_line=end)
    except Exception as e:
        print(f"Generic Exception for {obj=}")
        print(f"With {repo_root=}")
        print(e)
        return None

def github_link(repo_url: str, ref: str, src: SourceRef) -> str:
    return f"{repo_url.rstrip('/')}/blob/{ref}/{src.file_rel_repo}#L{src.start_line}-L{src.end_line}"


# ----------------------------
# Discovery / Inventory
# ----------------------------

@dataclasses.dataclass
class ObjectInfo:
    kind: str  # module|class|function|method|attribute
    full_name: str
    short_name: str
    module_name: str
    obj: Any

@dataclasses.dataclass
class ModuleInfo:
    name: str
    module: types.ModuleType
    classes: List[ObjectInfo]
    functions: List[ObjectInfo]
    attributes: List[ObjectInfo]
    child_modules: List[str]

def iter_package_modules(package_name: str) -> List[str]:
    pkg = importlib.import_module(package_name)
    pkg_path = safe_getattr(pkg, "__path__", None)
    if pkg_path is None:
        return [package_name]
    names = [package_name]
    for m in pkgutil.walk_packages(pkg_path, package_name + "."):
        names.append(m.name)
    return sorted(set(names))

def immediate_child_modules(parent: str, all_modules: List[str]) -> List[str]:
    """
    Return immediate children of parent.
    Example:
      parent="pkg.sub"
      candidates include "pkg.sub.a", "pkg.sub.a.b", "pkg.sub2.c"
      returns ["pkg.sub.a"] only.
    """
    prefix = parent + "."
    children = set()
    for m in all_modules:
        if not m.startswith(prefix):
            continue
        rest = m[len(prefix):]
        if not rest:
            continue
        # immediate child has no further dots in remainder
        if "." in rest:
            first = rest.split(".", 1)[0]
            children.add(prefix + first)
        else:
            children.add(prefix + rest)
    return sorted(children)

def is_defined_in_module(obj: Any, module_name: str) -> bool:
    return obj_module_name(obj) == module_name

def collect_module_info(
    module_name: str,
    include_private: bool = False,
    respect_all: bool = True,
    all_module_names: Optional[List[str]] = None,  # NEW
) -> ModuleInfo:
    mod = importlib.import_module(module_name)
    all_list = parse_all(mod) if respect_all else None
    exports = set(all_list) if all_list else None

    classes: List[ObjectInfo] = []
    functions: List[ObjectInfo] = []
    attributes: List[ObjectInfo] = []

    for name, val in vars(mod).items():
        if exports is not None and name not in exports:
            continue
        if not include_private and is_private_name(name):
            continue

        if inspect.isclass(val) and is_defined_in_module(val, module_name):
            classes.append(ObjectInfo("class", f"{module_name}.{name}", name, module_name, val))
        elif inspect.isfunction(val) and is_defined_in_module(val, module_name):
            functions.append(ObjectInfo("function", f"{module_name}.{name}", name, module_name, val))
        else:
            if isinstance(val, types.ModuleType):
                continue
            if (inspect.isclass(val) or inspect.isfunction(val)) and not is_defined_in_module(val, module_name):
                continue
            if exports is not None and name in exports:
                attributes.append(ObjectInfo("attribute", f"{module_name}.{name}", name, module_name, val))
            else:
                if isinstance(val, (str, int, float, bool, tuple, list, dict, set, frozenset, type(None))):
                    attributes.append(ObjectInfo("attribute", f"{module_name}.{name}", name, module_name, val))

    child_modules: List[str] = []
    if all_module_names:
        # Only meaningful for packages/subpackages, but harmless otherwise.
        child_modules = immediate_child_modules(module_name, all_module_names)

    return ModuleInfo(
        name=module_name,
        module=mod,
        classes=sorted(classes, key=lambda x: x.short_name.lower()),
        functions=sorted(functions, key=lambda x: x.short_name.lower()),
        attributes=sorted(attributes, key=lambda x: x.short_name.lower()),
        child_modules=child_modules,  # NEW
    )

def iter_init_modules(package_name: str, module_names: List[str]) -> List[str]:
    """
    Return the package and any subpackages' __init__ modules among module_names.
    """
    out = []
    for m in module_names:
        if m == package_name or m.endswith(".__init__"):
            # pkgutil.walk_packages usually yields packages by package name (e.g., pkg.subpkg),
            # not "pkg.subpkg.__init__". So we detect packages by importing and checking __path__.
            out.append(m)
    # For package modules (pkg.subpkg), import and treat that module as __init__.py for that package.
    # So we simply include all modules that are packages (have __path__).
    init_like = []
    for m in module_names:
        try:
            mod = importlib.import_module(m)
            if safe_getattr(mod, "__path__", None) is not None:
                init_like.append(m)
        except Exception:
            continue
    return sorted(set(out + init_like))

@dataclasses.dataclass
class AliasIndex:
    # object id -> list of alias full names (e.g., ["pkg.Foo", "pkg.sub.Foo"])
    aliases_by_id: Dict[int, List[str]] = dataclasses.field(default_factory=lambda: defaultdict(list))

    def add(self, obj: Any, alias_full_name: str) -> None:
        self.aliases_by_id[id(obj)].append(alias_full_name)

    def aliases_for(self, obj: Any) -> List[str]:
        return sorted(set(self.aliases_by_id.get(id(obj), [])))

def build_alias_index(
    package_name: str,
    module_names: List[str],
    include_private: bool,
    respect_all: bool,
) -> AliasIndex:
    """
    Scan package/subpackage modules (their __init__.py) for re-exported names.
    """
    alias = AliasIndex()
    init_modules = iter_init_modules(package_name, module_names)

    for mname in init_modules:
        try:
            mod = importlib.import_module(mname)
        except Exception:
            continue

        all_list = parse_all(mod) if respect_all else None
        if all_list is not None:
            export_names = list(all_list)
        else:
            export_names = [n for n in vars(mod).keys() if (include_private or not is_private_name(n))]

        for n in export_names:
            if not (include_private or not is_private_name(n)):
                continue
            val = safe_getattr(mod, n, None)
            if val is None:
                continue

            # Only alias objects that are part of this package (avoid stdlib etc.)
            vm = obj_module_name(val)
            if not vm.startswith(package_name):
                # But allow aliasing submodules imported as names:
                # e.g., import pkg.submodule as sub
                if isinstance(val, types.ModuleType) and safe_getattr(val, "__name__", "").startswith(package_name):
                    alias.add(val, f"{mname}.{n}" if mname != package_name else f"{package_name}.{n}")
                continue

            alias_full = f"{mname}.{n}" if mname != package_name else f"{package_name}.{n}"
            alias.add(val, alias_full)

    return alias

def preferred_name(package_name: str, obj: Any, default_full_name: str, alias_index: AliasIndex) -> str:
    aliases = alias_index.aliases_for(obj)
    if not aliases:
        return default_full_name

    def key(a: str) -> Tuple[int, int, str]:
        segs = a.split(".")
        # prefer fewer segments; then prefer top-level package alias; then lexicographic
        top_pref = 0 if a.count(".") == 1 and a.startswith(package_name + ".") else 1
        return (len(segs), top_pref, a)

    return sorted(aliases, key=key)[0]


# ----------------------------
# Rendering: MyST Markdown
# ----------------------------

def myst_target(full_name: str, source : bool) -> str:
    if source:
        return f"(api:{full_name})="
    else:
        return f"api:{full_name}"

def module_page_path(out_dir: Path, module_name: str) -> Path:
    return out_dir / "modules" / f"{slugify(module_name)}.md"

def class_page_path(out_dir: Path, class_full_name: str) -> Path:
    return out_dir / "classes" / f"{slugify(class_full_name)}.md"

def function_page_path(out_dir: Path, func_full_name: str) -> Path:
    return out_dir / "functions" / f"{slugify(func_full_name)}.md"

def link_to_module(out_dir: Path, module_name: str) -> str:
    #return module_page_path(out_dir, module_name).relative_to(out_dir).as_posix()
    return "#"+myst_target(module_name, source=False)

def link_to_class(out_dir: Path, class_full_name: str) -> str:
    #return class_page_path(out_dir, class_full_name).relative_to(out_dir).as_posix()
    return "#"+myst_target(class_full_name, source=False)

def link_to_function(out_dir: Path, func_full_name: str) -> str:
    # return function_page_path(out_dir, func_full_name).relative_to(out_dir).as_posix()
    return "#"+myst_target(func_full_name, source=False)

def render_object_header(full_name: str, title: str) -> str:
    return f"{myst_target(full_name, source=True)}\n# {title}\n\n"

def render_github_source(repo_url: str, ref: str, src: Optional[SourceRef]) -> str:
    if not (repo_url and ref and src):
        return ""
    return f"- Source: [GitHub]({github_link(repo_url, ref, src)})\n"

def render_signature_block(kind: str, name: str, sig: str) -> str:
    return f"```python\n{kind} {name}{sig}\n```\n\n"

def render_summary_table(rows: List[Tuple[str, str, str]]) -> str:
    if not rows:
        return ""
    out = ["| Name | Summary |", "|---|---|"]
    for name, link, summary in rows:
        nm = f"[`{name}`]({link})" if link else f"`{name}`"
        out.append(f"| {nm} | {summary} |")
    return "\n".join(out) + "\n\n"

def render_numpydoc(nd: NumpyDoc, extra_title_depth: int = 0) -> str:
    out: List[str] = []
    if nd.summary:
        out.append(nd.summary + "\n\n")
    if nd.extended_summary:
        out.append(nd.extended_summary + "\n\n")

    preferred = [
        "Parameters",
        "Other Parameters",
        "Returns",
        "Yields",
        "Raises",
        "Warns",
        "See Also",
        "Notes",
        "References",
        "Examples",
    ]
    used = set()

    def render_param_section(title: str, sec: NumpySection) -> str:
        lines: List[str] = [('#'*extra_title_depth) + f"## {title}\n\n"]
        if sec.params:
            for p in sec.params:
                head = f"- **{p.name}**"
                if p.type:
                    head += f" : *{p.type}*"
                lines.append(head)
                if p.desc:
                    lines.append(indent_md(p.desc, 2))
            lines.append("")
            return "\n".join(lines) + "\n"
        body = sec.body.strip()
        if body:
            return ('#'*extra_title_depth) + f"## {title}\n\n{body}\n\n"
        return ""

    def render_text_section(title: str, sec: NumpySection) -> str:
        body = sec.body.strip()
        if not body:
            return ""
        return ('#'*extra_title_depth) + f"## {title}\n\n{body}\n\n"

    for t in preferred:
        sec = nd.sections.get(t)
        if not sec:
            continue
        used.add(t)
        if t in ("Parameters", "Other Parameters", "Returns", "Yields", "Raises", "Warns", "See Also"):
            out.append(render_param_section(t, sec))
        else:
            out.append(render_text_section(t, sec))

    for t, sec in nd.sections.items():
        if t in used:
            continue
        if sec.params:
            out.append(render_param_section(t, sec))
        else:
            out.append(render_text_section(t, sec))
    return "".join(out)


def inheritance_mermaid_flowchart(cls, direction="TB", include_object=False):
    """
    Render a Mermaid flowchart showing the inheritance DAG of `cls`.
    Uses edges: Base --> Derived (so roots at top by default with TB).

    Parameters
    ----------
    cls : type
        Class to render.
    direction : str
        Mermaid flowchart direction: "TB", "BT", "LR", "RL".
    include_object : bool
        Whether to include `object` in the diagram.
    """

    def node_id(c):
        # stable-ish id; avoids collisions for same __name__ in different modules
        return f"C{abs(hash((c.__module__, c.__qualname__))) }"

    def node_label(c):
        qn = c.__qualname__
        mod = c.__module__
        return f"{mod}.{qn}" if mod not in ("builtins",) else qn

    nodes = {}  # id -> label
    edges = set()  # (base_id, derived_id)
    visited = set()

    def walk(c):
        if c in visited:
            return
        visited.add(c)

        if (c is object) and not include_object:
            return

        cid = node_id(c)
        nodes[cid] = node_label(c)

        for b in getattr(c, "__bases__", ()):
            if (b is object) and not include_object:
                continue

            bid = node_id(b)
            nodes[bid] = node_label(b)

            # base -> derived
            edges.add((bid, cid))

            walk(b)

    walk(cls)

    # Build Mermaid text
    lines = [f"flowchart {direction}"]

    # Declare nodes (quote labels to handle dots, etc.)
    for nid, label in sorted(nodes.items(), key=lambda x: x[1]):
        lines.append(f'  {nid}["{label}"]')

    # Declare edges
    for a, b in sorted(edges, key=lambda e: (nodes.get(e[0], ""), nodes.get(e[1], ""))):
        lines.append(f"  {a} --> {b}")

    return "\n".join(lines)


def render_mermaid_inheritance(cls, direction="TB", include_object=False) -> str:
    """
    Render a Mermaid flowchart showing the inheritance DAG of `cls`.
    Uses edges: Base --> Derived (so roots at top by default with TB).

    Parameters
    ----------
    cls : type
        Class to render.
    direction : str
        Mermaid flowchart direction: "TB", "BT", "LR", "RL".
    include_object : bool
        Whether to include `object` in the diagram.
    """

    def node_id(c):
        # stable-ish id; avoids collisions for same __name__ in different modules
        return f"C{abs(hash((c.__module__, c.__qualname__))) }"

    def node_label(c):
        qn = c.__qualname__
        mod = c.__module__
        return f"{mod}.{qn}" if mod not in ("builtins",) else qn

    nodes = {}  # id -> label
    edges = set()  # (base_id, derived_id)
    visited = set()

    def walk(c):
        if c in visited:
            return
        visited.add(c)

        if (c is object) and not include_object:
            return

        cid = node_id(c)
        nodes[cid] = node_label(c)

        for b in getattr(c, "__bases__", ()):
            if (b is object) and not include_object:
                continue

            bid = node_id(b)
            nodes[bid] = node_label(b)

            # base -> derived
            edges.add((bid, cid))

            walk(b)

    walk(cls)

    # Build Mermaid text
    lines = ["```{mermaid}", f"flowchart {direction}"]

    # Declare nodes (quote labels to handle dots, etc.)
    for nid, label in sorted(nodes.items(), key=lambda x: x[1]):
        lines.append(f'  {nid}["{label}"]')

    # Declare edges
    for a, b in sorted(edges, key=lambda e: (nodes.get(e[0], ""), nodes.get(e[1], ""))):
        lines.append(f"  {a} --> {b}")
    lines.append("```")
    return "\n".join(lines) + "\n\n"


def render_parent_links(cls: type, out_dir: Path, package_prefix: str, alias_index: "AliasIndex") -> str:
    bases = [b for b in cls.__mro__[1:] if b is not object]
    if not bases:
        return ""
    items: List[str] = []
    for b in bases:
        b_mod = obj_module_name(b)
        b_default = f"{b_mod}.{qualname(b)}"
        b_pref = preferred_name(package_prefix, b, b_default, alias_index)
        if b_mod.startswith(package_prefix):
            items.append(f"- Parent: [`{b_pref}`]({link_to_class(out_dir, b_pref)})")
        else:
            items.append(f"- Parent: `{b_pref}`")
    return "\n".join(items) + "\n\n"

def iter_class_members(cls: type, include_private: bool) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Any]]]:
    methods: List[Tuple[str, Any]] = []
    attrs: List[Tuple[str, Any]] = []

    for name, val in cls.__dict__.items():
        if not include_private and is_private_name(name):
            continue
        func = None
        if inspect.isfunction(val):
            func = val
        elif isinstance(val, (staticmethod, classmethod)):
            func = val.__func__
        if func is not None:
            methods.append((name, func))
        elif isinstance(val, property):
            attrs.append((name, val))
        else:
            if isinstance(val, (str, int, float, bool, tuple, list, dict, set, frozenset, type(None))):
                attrs.append((name, val))

    methods.sort(key=lambda x: x[0].lower())
    attrs.sort(key=lambda x: x[0].lower())
    return methods, attrs


# ----------------------------
# Page renderers (with alias-preferred names)
# ----------------------------

def render_module_page(
    out_dir: Path,
    minfo: ModuleInfo,
    repo_url: str,
    ref: str,
    repo_root: Optional[Path],
    package_name: str,
    alias_index: AliasIndex,
) -> str:
    mod = minfo.module
    out: List[str] = ["---\n", f"short_title: {minfo.name.split('.')[-1]}\n", "---\n"]
    out.append(render_object_header(minfo.name, minfo.name))
    out.append(render_github_source(repo_url, ref, get_source_ref(mod, repo_root)))
    out.append("\n")

    out.append(render_numpydoc(parse_numpydoc(get_doc(mod))))

    # NEW: Submodules/Subpackages navigation
    if minfo.child_modules:
        out.append("## Submodules\n\n")
        rows: List[Tuple[str, str, str]] = []
        for sm in minfo.child_modules:
            # display short leaf name
            leaf = sm.split(".")[-1]
            # try to pull first-line summary from that module's docstring (best-effort)
            summary = ""
            try:
                smod = importlib.import_module(sm)
                summary = first_sentence_or_line(get_doc(smod))
            except Exception:
                summary = ""
            rows.append((leaf, link_to_module(out_dir, sm), summary))
        out.append(render_summary_table(rows))

    if minfo.classes:
        out.append("## Classes\n\n")
        rows: List[Tuple[str, str, str]] = []
        for c in minfo.classes:
            c_pref = preferred_name(package_name, c.obj, c.full_name, alias_index)
            rows.append((c_pref.split(".")[-1], link_to_class(out_dir, c_pref), first_sentence_or_line(get_doc(c.obj))))
        out.append(render_summary_table(rows))

    if minfo.functions:
        out.append("## Functions\n\n")
        rows = []
        for f in minfo.functions:
            f_pref = preferred_name(package_name, f.obj, f.full_name, alias_index)
            rows.append((f_pref.split(".")[-1], link_to_function(out_dir, f_pref), first_sentence_or_line(get_doc(f.obj))))
        out.append(render_summary_table(rows))

    return "".join(out)

def render_function_page(
    out_dir: Path,
    finfo: ObjectInfo,
    repo_url: str,
    ref: str,
    repo_root: Optional[Path],
    package_name: str,
    alias_index: AliasIndex,
) -> str:
    fn = finfo.obj
    default = finfo.full_name
    pref = preferred_name(package_name, fn, default, alias_index)
    aliases = [a for a in alias_index.aliases_for(fn) if a != pref]

    out: List[str] = ["---\n", f"short_title: {pref.split('.')[-1]}\n", "---\n"]
    out.append(render_object_header(pref, pref))

    # Show definition + aliases
    if pref != default:
        out.append(f"- Defined as: `{default}`\n")
    if aliases:
        out.append("- Also available as:\n")
        for a in aliases:
            out.append(f"  - `{a}`\n")
    out.append(f"- Module: [`{finfo.module_name}`]({link_to_module(out_dir, finfo.module_name)})\n")
    out.append(render_github_source(repo_url, ref, get_source_ref(fn, repo_root)))
    out.append("\n")

    out.append("## Signature\n\n")
    out.append(render_signature_block("def", pref, format_signature(fn)))
    out.append(render_numpydoc(parse_numpydoc(get_doc(fn))))

    return "".join(out)

def render_class_page(
    out_dir: Path,
    cinfo: ObjectInfo,
    repo_url: str,
    ref: str,
    repo_root: Optional[Path],
    package_name: str,
    alias_index: AliasIndex,
    include_private: bool,
) -> str:
    cls = cinfo.obj
    default = cinfo.full_name
    pref = preferred_name(package_name, cls, default, alias_index)
    aliases = [a for a in alias_index.aliases_for(cls) if a != pref]

    out: List[str] = ["---\n", f"short_title: {pref.split('.')[-1]}\n", "---\n"]
    out.append(render_object_header(pref, pref))

    if pref != default:
        out.append(f"- Defined as: `{default}`\n")
    if aliases:
        out.append("- Also available as:\n")
        for a in aliases:
            out.append(f"  - `{a}`\n")
    out.append(f"- Module: [`{cinfo.module_name}`]({link_to_module(out_dir, cinfo.module_name)})\n")
    out.append(render_github_source(repo_url, ref, get_source_ref(cls, repo_root)))
    out.append("\n")

    out.append(render_parent_links(cls, out_dir, package_name, alias_index))
    out.append(render_mermaid_inheritance(cls))

    out.append("## Signature\n\n")
    out.append(render_signature_block("class", pref, format_signature(cls)))
    out.append(render_numpydoc(parse_numpydoc(get_doc(cls))))

    methods, attrs = iter_class_members(cls, include_private=include_private)

    if attrs:
        out.append("## Attributes\n\n")
        rows: List[Tuple[str, str, str]] = []
        for name, obj in attrs:
            full = f"{pref}.{name}"
            rows.append((name, f"#{myst_target(full, source=False)}", first_sentence_or_line(get_doc(obj))))
        out.append(render_summary_table(rows))

        for name, obj in attrs:
            full = f"{pref}.{name}"
            out.append(f"{myst_target(full, source=True)}\n")
            out.append(f"### `{name}`\n\n")
            if isinstance(obj, property):
                out.append(render_numpydoc(parse_numpydoc(inspect.getdoc(obj.fget) or ""), extra_title_depth=2))

    if methods:
        out.append("## Methods\n\n")
        rows = []
        for name, func in methods:
            full = f"{pref}.{name}"
            rows.append((name, f"#{myst_target(full, source=False)}", first_sentence_or_line(get_doc(func))))
        out.append(render_summary_table(rows))

        for name, func in methods:
            full = f"{pref}.{name}"
            out.append(f"{myst_target(full, source=True)}\n")
            out.append(f"### `{name}`\n\n")
            out.append(render_github_source(repo_url, ref, get_source_ref(func, repo_root)))
            out.append("\n")
            out.append(render_signature_block("def", full, format_signature(func)))
            out.append(render_numpydoc(parse_numpydoc(get_doc(func)), extra_title_depth=2))

    return "".join(out)

def render_index_page(package_name: str, module_names: List[str]) -> str:
    out: List[str] = []
    out.append(f"# API Reference: `{package_name}`\n\n")
    out.append("```{toctree}\n:maxdepth: 2\n:caption: Modules\n\n")
    for m in module_names:
        out.append(f"modules/{slugify(m)}\n")
    out.append("```\n")
    return "".join(out)


# ----------------------------
# Main
# ----------------------------

def write_file(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate MyST Markdown API docs from a Python package (Jupyter Book v2).")
    ap.add_argument("--package", required=True, help="Top-level importable package name, e.g. mypkg")
    ap.add_argument("--out", required=True, help="Output directory for .md files, e.g. docs/api")
    ap.add_argument("--repo", default="", help="GitHub repo URL, e.g. https://github.com/ORG/REPO")
    ap.add_argument("--ref", default="main", help="Git ref for links, e.g. main or a tag")
    ap.add_argument("--repo-root", default="", help="Local filesystem path to repo root for source mapping (repo root, not src/)")
    ap.add_argument("--include-private", action="store_true", help="Include private members (leading underscore)")
    ap.add_argument("--no-respect-all", action="store_true", help="Ignore __all__ and scan all public names")
    ap.add_argument("--no-index", action="store_true", help="Do not generate an index.md")
    args = ap.parse_args(argv)

    package_name = args.package
    out_dir = Path(args.out).resolve()
    repo_url = args.repo
    ref = args.ref
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None

    ensure_dir(out_dir)
    ensure_dir(out_dir / "modules")
    ensure_dir(out_dir / "classes")
    ensure_dir(out_dir / "functions")

    module_names = iter_package_modules(package_name)

    # Alias index from __init__.py modules (package + subpackages)
    alias_index = build_alias_index(
        package_name=package_name,
        module_names=module_names,
        include_private=args.include_private,
        respect_all=not args.no_respect_all,
    )

    module_infos: Dict[str, ModuleInfo] = {}
    class_infos: List[ObjectInfo] = []
    func_infos: List[ObjectInfo] = []

    for m in module_names:
        try:
            minfo = collect_module_info(
                m,
                include_private=args.include_private,
                respect_all=not args.no_respect_all,
                all_module_names=module_names,
            )
            module_infos[m] = minfo
            class_infos.extend(minfo.classes)
            func_infos.extend(minfo.functions)
        except Exception as e:
            content = render_object_header(m, m)
            content += f"**Error importing/inspecting module** `{m}`:\n\n```text\n{type(e).__name__}: {e}\n```\n"
            write_file(module_page_path(out_dir, m), content)

    # Module pages (module names are always their real module path)
    for m, minfo in module_infos.items():
        content = render_module_page(out_dir, minfo, repo_url, ref, repo_root, package_name, alias_index)
        write_file(module_page_path(out_dir, m), content)

    # Function pages (module-level only) with preferred alias paths
    written_funcs = set()
    for f in func_infos:
        try:
            pref = preferred_name(package_name, f.obj, f.full_name, alias_index)
            if pref in written_funcs:
                continue
            written_funcs.add(pref)
            content = render_function_page(out_dir, f, repo_url, ref, repo_root, package_name, alias_index)
            write_file(function_page_path(out_dir, pref), content)
        except Exception as e:
            nm = f.full_name
            content = render_object_header(nm, nm)
            content += f"**Error inspecting function** `{nm}`:\n\n```text\n{type(e).__name__}: {e}\n```\n"
            write_file(function_page_path(out_dir, nm), content)

    # Class pages with preferred alias paths
    written_classes = set()
    for c in class_infos:
        try:
            pref = preferred_name(package_name, c.obj, c.full_name, alias_index)
            if pref in written_classes:
                continue
            written_classes.add(pref)
            content = render_class_page(out_dir, c, repo_url, ref, repo_root, package_name, alias_index, args.include_private)
            write_file(class_page_path(out_dir, pref), content)
        except Exception as e:
            nm = c.full_name
            content = render_object_header(nm, nm)
            content += f"**Error inspecting class** `{nm}`:\n\n```text\n{type(e).__name__}: {e}\n```\n"
            write_file(class_page_path(out_dir, nm), content)

    if not args.no_index:
        write_file(out_dir / "index.md", render_index_page(package_name, module_names))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
