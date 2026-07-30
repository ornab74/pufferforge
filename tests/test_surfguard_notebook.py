from __future__ import annotations

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "SurfGuard_USA_PufferForge_Colab.ipynb"


def _notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def _code_source():
    return "\n\n".join(cell.source for cell in _notebook().cells if cell.cell_type == "code")


def test_notebook_json_and_all_code_cells_compile() -> None:
    notebook = _notebook()
    assert notebook.nbformat == 4
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"SurfGuard cell {index}", "exec")
            assert not any(output.get("output_type") == "error" for output in cell.get("outputs", []))


def test_notebook_uses_public_pufferforge_api_and_portable_models() -> None:
    source = _code_source()
    assert "from pufferforge.pufferforge" not in source
    assert "from pufferforge import PPOTrainer, PythonVectorEnv, TrainConfig" in source
    assert '"gpt-5.6-terra"' not in source
    assert "SURFGUARD_OPENAI_MODEL" in source


def test_coops_requests_are_chunked_to_service_limit() -> None:
    source = _code_source()
    assert "COOPS_MAX_REQUEST_DAYS = 31" in source
    assert "max_days=COOPS_MAX_REQUEST_DAYS" in source
    assert "_coops_data_single" in source


def test_notebook_has_no_stale_saved_outputs() -> None:
    for cell in _notebook().cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []


def test_ndbc_sentinels_and_optional_grib_are_explicit() -> None:
    source = _code_source()
    assert '"WSPD": 99.0' in source
    assert '"PRES": 9999.0' in source
    assert "RUN_OPERATIONAL_GFSWAVE=False" in source.replace(" ", "")
    assert "requires optional packages cfgrib and eccodes" in source


def _function_source(name: str) -> str:
    for cell in _notebook().cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return ast.get_source_segment(cell.source, node)
    raise AssertionError(f"function not found: {name}")


def _class_source(name: str) -> str:
    for cell in _notebook().cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return ast.get_source_segment(cell.source, node)
    raise AssertionError(f"class not found: {name}")


def test_supervised_training_drops_empty_features_and_preserves_holdout_positives() -> None:
    source = _code_source()
    assert "ALL_MISSING_FEATURE_COLUMNS" in source
    assert "ACTIVE_FEATURE_COLUMNS" in source
    assert "keep_empty_features=True" in source.replace(" ", "")
    assert "event_aware_temporal_split" in source
    assert '.groupby(["station_id", "month"], group_keys=False)\n        .apply(' not in source

    import numpy as np
    import pandas as pd

    namespace = {"np": np, "pd": pd, "LABEL_WINDOW_HOURS": 3}
    exec(_function_source("_three_way_counts"), namespace)
    exec(_function_source("event_aware_temporal_split"), namespace)

    negatives = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=300, freq="12h", tz="UTC"),
        "station_id": ["N"] * 300,
        "danger_label": np.zeros(300, dtype=int),
    })
    episodes = []
    for index, start in enumerate(pd.to_datetime([
        "2020-02-01", "2020-04-01", "2020-06-01",
        "2020-08-01", "2020-10-01", "2020-12-01",
    ], utc=True)):
        episodes.append(pd.DataFrame({
            "time": pd.date_range(start, periods=7, freq="h"),
            "station_id": [f"P{index % 2}"] * 7,
            "danger_label": np.ones(7, dtype=int),
        }))
    frame = pd.concat([negatives, *episodes], ignore_index=True)
    train, valid, test, summary = namespace["event_aware_temporal_split"](frame)
    assert summary["positive_event_groups"] == 6
    assert train["danger_label"].nunique() == 2
    assert valid["danger_label"].nunique() == 2
    assert test["danger_label"].nunique() == 2


def test_notebook_beach_alert_env_runs_with_pufferforge_ppo() -> None:
    import numpy as np
    from pufferforge import PPOTrainer, PythonVectorEnv, TrainConfig

    features = ["a", "b", "c"]
    rng = np.random.default_rng(3)
    namespace = {
        "np": np,
        "RL_FEATURES": features,
        "rl_obs": rng.normal(size=(64, len(features))).astype(np.float32),
        "rl_labels": np.asarray([0, 1] * 32, dtype=np.int8),
        "rl_severity": np.ones(64, dtype=np.float32),
    }
    exec(_class_source("BeachAlertEnv"), namespace)
    cls = namespace["BeachAlertEnv"]
    env = PythonVectorEnv([lambda i=i: cls(seed=i, episode_length=8) for i in range(2)], seed=1)
    config = TrainConfig(seed=1, total_timesteps=16, num_envs=2, horizon=4,
        minibatch_size=8, update_epochs=1, hidden_size=8, hidden_layers=1,
        checkpoint_interval=0)
    trainer = PPOTrainer(env, config)
    history = trainer.train()
    trainer.close()
    assert history and history[-1].global_step == 16
