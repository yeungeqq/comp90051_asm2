"""Exercise Kaggle source discovery with flat datasets and source archives."""

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


LAUNCHER = json.loads((Path(__file__).resolve().parents[1] / "kaggle_runner.ipynb").read_text())


def make_project(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training").mkdir()
    (directory / "training" / "protocol.py").write_text("")
    (directory / "config.py").write_text("")
    (directory / "main.py").write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path(__file__).with_name('launch_arguments.json').write_text(json.dumps(sys.argv[1:]))\n")
    return directory


def run_setup(input_root, project):
    source = "".join(LAUNCHER["cells"][1]["source"])
    source = source.replace("KAGGLE_INPUT_ROOT = Path('/kaggle/input')",
                            f"KAGGLE_INPUT_ROOT = Path({str(input_root)!r})")
    source = source.replace("PROJECT = Path('/kaggle/working/comp90051-amazon-anomaly')",
                            f"PROJECT = Path({str(project)!r})")
    namespace = {}
    exec(compile(source, "kaggle_runner_setup", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("layout", ["flat", "long_mount", "wrapped", "flat_zip", "wrapped_zip"])
def test_source_layouts_launch_from_writable_project(tmp_path, layout):
    input_root = tmp_path / "input"
    dataset = input_root / "program"
    destination = tmp_path / "working" / "comp90051-amazon-anomaly"
    destination.mkdir(parents=True)
    # Re-running setup must retain results already saved in writable storage.
    (destination / "saved_result.txt").write_text("retain")
    if layout in {"flat_zip", "wrapped_zip"}:
        source = make_project(tmp_path / "source")
        dataset.mkdir(parents=True)
        name = "project.zip" if layout == "flat_zip" else "comp90051-amazon-anomaly.zip"
        prefix = Path(".") if layout == "flat_zip" else Path("comp90051-amazon-anomaly")
        with zipfile.ZipFile(dataset / name, "w") as archive:
            for path in source.rglob("*.py"):
                archive.write(path, prefix / path.relative_to(source))
    else:
        if layout == "long_mount":
            dataset = input_root / "datasets" / "account" / "program"
        elif layout == "wrapped":
            dataset = dataset / "comp90051-amazon-anomaly"
        make_project(dataset)
        (dataset / "__MACOSX").mkdir()
        (dataset / "__MACOSX" / "._main.py").write_text("metadata")

    namespace = run_setup(input_root, destination)
    assert (destination / "main.py").is_file()
    assert (destination / "saved_result.txt").read_text() == "retain"
    assert not (destination / "__MACOSX").exists()
    exec(compile("".join(LAUNCHER["cells"][3]["source"]), "kaggle_runner_configuration", "exec"), namespace)
    arguments = json.loads((destination / "launch_arguments.json").read_text())
    assert arguments == ["--input-root", str(input_root), "--output-root", str(destination / "outputs"),
                         "--recompute-features", "--dry-run"]


def test_program_dataset_is_preferred(tmp_path):
    input_root = tmp_path / "input"
    program = make_project(input_root / "program")
    make_project(input_root / "other-project")
    namespace = run_setup(input_root, tmp_path / "working")
    assert namespace["SOURCE_PROJECT"] == program


def test_ambiguous_source_requires_explicit_path(tmp_path):
    input_root = tmp_path / "input"
    make_project(input_root / "first")
    make_project(input_root / "second")
    with pytest.raises(ValueError, match="Set SOURCE_PROJECT explicitly"):
        run_setup(input_root, tmp_path / "working")


def test_source_zip_cannot_escape_extraction_directory(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    with zipfile.ZipFile(input_root / "project.zip", "w") as archive:
        archive.writestr("../outside.py", "invalid")
    with pytest.raises(ValueError, match="outside its extraction directory"):
        run_setup(input_root, tmp_path / "working")


@pytest.mark.parametrize("fail", [False, True])
def test_experiment_cell_streams_and_saves_child_errors(tmp_path, capsys, fail):
    child = "print('training progress', flush=True)"
    if fail:
        child += "; raise ValueError('specific child failure')"
    namespace = {"OUTPUT_ROOT": tmp_path / "outputs", "subprocess": subprocess,
                 "json": json, "command": [sys.executable, "-u", "-c", child]}
    source = compile("".join(LAUNCHER["cells"][5]["source"]), "kaggle_runner_experiment", "exec")
    if fail:
        with pytest.raises(RuntimeError, match="ValueError: specific child failure"):
            exec(source, namespace)
    else:
        exec(source, namespace)
    logged = (namespace["OUTPUT_ROOT"] / "run.log").read_text()
    displayed = capsys.readouterr().out
    assert "training progress" in logged and "training progress" in displayed
    if fail:
        assert "Traceback" in logged and "ValueError: specific child failure" in logged
        assert "ValueError: specific child failure" in displayed
    else:
        assert "Experiment finished successfully" in displayed
