import importlib

from reglens import paths


def test_reglens_home_overrides_root(monkeypatch, tmp_path):
    monkeypatch.setenv("REGLENS_HOME", str(tmp_path))
    assert paths.project_root() == tmp_path.resolve()


def test_falls_back_to_cwd_outside_a_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("REGLENS_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    fake_module = tmp_path / "site-packages" / "reglens" / "paths.py"
    fake_module.parent.mkdir(parents=True)
    monkeypatch.setattr(paths, "__file__", str(fake_module))
    assert paths.project_root() == tmp_path
    importlib.reload(paths)  # restore the real module state
