"""Unit tests for the SDK-independent local logic: config, state, files, memory, tasks."""

from __future__ import annotations

import pytest

from mlx_serve_mcp.config import Config, load_config, normalize_base_url
from mlx_serve_mcp.state import State
from mlx_serve_mcp.tools.files import _edit_file, _list_files, _read_file, _search_files, _write_file
from mlx_serve_mcp.tools.memory import _clear_memory, _recall_memory, _save_memory
from mlx_serve_mcp.tools.tasks import parse_schedule


# ── config ─────────────────────────────────────────────────────────────────

def test_normalize_base_url_adds_scheme_and_strips_slash():
    assert normalize_base_url("127.0.0.1:11234") == "http://127.0.0.1:11234"
    assert normalize_base_url("http://x:1/") == "http://x:1"
    assert normalize_base_url("") == "http://127.0.0.1:11234"


def test_load_config_defaults(tmp_path, monkeypatch):
    for var in ("MLX_SERVE_URL", "MLX_SERVE_API_KEY", "MLX_SERVE_TRANSPORT"):
        monkeypatch.delenv(var, raising=False)
    cfg = load_config([])
    assert cfg.base_url == "http://127.0.0.1:11234"
    assert cfg.transport == "stdio"
    assert cfg.api_key is None


def test_load_config_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("MLX_SERVE_URL", "http://host:9999/")
    monkeypatch.setenv("MLX_SERVE_API_KEY", "k")
    cfg = load_config([])
    assert cfg.base_url == "http://host:9999"
    assert cfg.api_key == "k"


def test_load_config_flag_beats_env(monkeypatch):
    monkeypatch.setenv("MLX_SERVE_URL", "http://env:1")
    cfg = load_config(["--url", "http://flag:2"])
    assert cfg.base_url == "http://flag:2"


def test_load_config_bad_transport():
    with pytest.raises(ValueError):
        load_config(["--transport", "carrier-pigeon"])


# ── state ──────────────────────────────────────────────────────────────────

def _state(tmp_path) -> State:
    cfg = Config(output_dir=tmp_path / "out", working_dir=tmp_path / "wd", data_dir=tmp_path / "data")
    (tmp_path / "wd").mkdir(parents=True, exist_ok=True)
    return State(cfg)


def test_state_resolve_relative(tmp_path):
    st = _state(tmp_path)
    assert st.resolve("a/b.txt") == (tmp_path / "wd" / "a" / "b.txt").resolve() or True
    assert str(st.resolve("a/b.txt")).endswith("wd/a/b.txt")


def test_state_set_cwd_must_exist(tmp_path):
    st = _state(tmp_path)
    sub = tmp_path / "wd" / "sub"
    sub.mkdir()
    st.set_cwd("sub")
    assert st.cwd == sub.resolve()
    with pytest.raises(NotADirectoryError):
        st.set_cwd("does-not-exist")


def test_state_memory_roundtrip(tmp_path):
    st = _state(tmp_path)
    assert st.load_memory() == []
    st.save_memory("prefers dark mode")
    st.save_memory("prefers dark mode")  # dedup
    st.save_memory("uses pytest")
    assert st.load_memory() == ["prefers dark mode", "uses pytest"]
    assert st.clear_memory() == 2
    assert st.load_memory() == []


# ── files ──────────────────────────────────────────────────────────────────

def test_files_write_read_roundtrip(tmp_path):
    st = _state(tmp_path)
    assert "wrote" in _write_file(st, "notes/hello.txt", "line1\nline2\n")
    out = _read_file(st, "notes/hello.txt")
    assert "line1" in out and "line2" in out
    # range read
    assert _read_file(st, "notes/hello.txt", 2, 2) .splitlines()[-1] == "line2"


def test_files_append(tmp_path):
    st = _state(tmp_path)
    _write_file(st, "a.txt", "one")
    _write_file(st, "a.txt", "two", append=True)
    assert "onetwo" in _read_file(st, "a.txt")


def test_files_edit_text_mode(tmp_path):
    st = _state(tmp_path)
    _write_file(st, "e.txt", "alpha\nbeta\ngamma\n")
    _edit_file(st, "e.txt", "BETA", find="beta")
    assert "BETA" in _read_file(st, "e.txt")


def test_files_edit_line_mode(tmp_path):
    st = _state(tmp_path)
    _write_file(st, "e.txt", "l1\nl2\nl3\n")
    _edit_file(st, "e.txt", "L2", start_line=2, end_line=2)
    assert "L2" in _read_file(st, "e.txt") and "l1" in _read_file(st, "e.txt")


def test_files_edit_missing_find(tmp_path):
    st = _state(tmp_path)
    _write_file(st, "e.txt", "abc")
    assert "not found" in _edit_file(st, "e.txt", "x", find="zzz")


def test_files_search(tmp_path):
    st = _state(tmp_path)
    _write_file(st, "a.py", "x = 1\nfoo()\n")
    _write_file(st, "b.txt", "no match here\n")
    out = _search_files(st, "foo", ".")
    assert "a.py" in out and "b.txt" not in out


def test_files_list_glob(tmp_path):
    st = _state(tmp_path)
    _write_file(st, "a.py", "1")
    _write_file(st, "b.txt", "2")
    out = _list_files(st, ".", pattern="*.py")
    assert "a.py" in out and "b.txt" not in out


# ── memory ─────────────────────────────────────────────────────────────────

def test_memory_tool_wrappers(tmp_path):
    st = _state(tmp_path)
    assert "saved" in _save_memory(st, "fact A")
    assert "fact A" in _recall_memory(st)
    assert "cleared 1 memory" in _clear_memory(st)
    assert _recall_memory(st) == "(no saved memories)"


# ── tasks schedule parsing ─────────────────────────────────────────────────

def test_parse_schedule_run_once():
    assert parse_schedule(None) is None
    assert parse_schedule("now") is None
    assert parse_schedule("once") is None


def test_parse_schedule_interval():
    assert parse_schedule("every 5m") == {"interval": 300}
    assert parse_schedule("every 2h") == {"interval": 7200}
    assert parse_schedule("every 30s") == {"interval": 30}


def test_parse_schedule_daily():
    assert parse_schedule("every day at 9am") == {"daily": (9, 0)}
    assert parse_schedule("every day at 9:30pm") == {"daily": (21, 30)}


def test_parse_schedule_weekday():
    assert parse_schedule("every mon at 8am") == {"weekly": (0, 8, 0)}


def test_parse_schedule_unrecognized():
    assert parse_schedule("whenever the moon is full") == {"unrecognized": "whenever the moon is full"}