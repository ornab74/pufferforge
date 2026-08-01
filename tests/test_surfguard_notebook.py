from __future__ import annotations

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "SurfGuard_USA_PufferForge_Colab.ipynb"


def _notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def _code_source():
    return "\n\n".join(cell.source for cell in _notebook().cells if cell.cell_type == "code")


def test_notebook_json_and_all_code_cells_compile() -> None:
    notebook = _notebook()
    assert notebook.nbformat == 4
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"{NOTEBOOK.name} cell {index}", "exec")
            assert not any(
                output.get("output_type") == "error"
                for output in cell.get("outputs", [])
            )


def test_notebook_uses_public_pufferforge_api_and_portable_models() -> None:
    source = _code_source()
    assert "from pufferforge.pufferforge" not in source
    assert "from pufferforge import PPOTrainer, PythonVectorEnv, TrainConfig" in source
    assert '"gpt-5.6-terra"' not in source
    assert 'SURFGUARD_OPENAI_MODEL' in source


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


def test_date_chunk_helper_precedes_coops_use_and_chunks_a_year() -> None:
    import time

    import pandas as pd

    notebook = _notebook()
    date_index = next(i for i, c in enumerate(notebook.cells) if c.cell_type == "code" and "def date_chunks" in c.source)
    coops_index = next(i for i, c in enumerate(notebook.cells) if c.cell_type == "code" and "def coops_data" in c.source)
    assert date_index < coops_index

    namespace = {"pd": pd, "time": time, "COOPS_MAX_REQUEST_DAYS": 31}
    exec(_function_source("date_chunks"), namespace)
    calls = []

    def fake_single(station_id, begin_date, end_date, product, **kwargs):
        calls.append((begin_date, end_date))
        return pd.DataFrame({"time": [pd.Timestamp(begin_date, tz="UTC")], "value": [1.0]})

    namespace["_coops_data_single"] = fake_single
    exec(_function_source("coops_data"), namespace)
    result = namespace["coops_data"]("123", "20200101", "20201231", "predictions")
    assert not result.empty
    assert len(calls) == 12
    for begin, end in calls:
        assert (pd.Timestamp(end) - pd.Timestamp(begin)).days <= 30


def test_ndbc_parser_converts_documented_nines_to_nan() -> None:
    import gzip
    import io

    import numpy as np
    import pandas as pd

    namespace = {"gzip": gzip, "io": io, "np": np, "pd": pd}
    exec(_function_source("parse_ndbc_stdmet"), namespace)
    text = """#YYYY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS TIDE
#yr mo dy hr mn degT m/s m/s m sec sec degT hPa degC degC degC nmi ft
2025 01 01 00 00 999 99.0 99.0 99.0 99.0 99.0 999 9999.0 999.0 999.0 999.0 99.0 99.0
2025 01 01 01 00 180 5.0 6.0 1.2 8.0 7.0 190 1014.0 20.0 21.0 18.0 10.0 1.2
"""
    frame = namespace["parse_ndbc_stdmet"](gzip.compress(text.encode()), "41001")
    assert frame.iloc[0]["WSPD"] != frame.iloc[0]["WSPD"]
    assert frame.iloc[0]["PRES"] != frame.iloc[0]["PRES"]
    assert frame.iloc[1]["WVHT"] == 1.2



def test_supervised_training_drops_empty_features_and_preserves_holdout_positives() -> None:
    source = _code_source()
    assert "ALL_MISSING_FEATURE_COLUMNS" in source
    assert "ACTIVE_FEATURE_COLUMNS" in source
    assert "FiniteMedianImputer" in source
    assert "event_aware_temporal_split" in source
    assert ".groupby([\"station_id\", \"month\"], group_keys=False)\n        .apply(" not in source

    import numpy as np
    import pandas as pd

    namespace = {
        "np": np,
        "pd": pd,
        "LABEL_WINDOW_HOURS": 3,
    }
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
    assert valid["danger_label"].sum() > 0
    assert test["danger_label"].sum() > 0

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
    config = TrainConfig(
        seed=1, total_timesteps=16, num_envs=2, horizon=4,
        minibatch_size=8, update_epochs=1, hidden_size=8, hidden_layers=1,
        checkpoint_interval=0,
    )
    trainer = PPOTrainer(env, config)
    history = trainer.train()
    trainer.close()
    assert history and history[-1].global_step == 16


def test_notebook_enables_confidence_aware_ppo() -> None:
    source = _code_source()
    assert "gae_ensemble=((0.97, 0.90), (0.995, 0.97))" in source
    assert "value_heads=3" in source
    assert "critic_bootstrap_probability=0.8" in source
    assert "uncertainty_coef=1.0" in source

def test_climatology_skips_empty_features_without_runtime_warnings(tmp_path) -> None:
    import warnings

    import numpy as np
    import pandas as pd

    namespace = {
        "np": np,
        "pd": pd,
        "CLIMATOLOGY_FEATURES": ["tide_obs_m", "surge_residual_m", "WVHT"],
        "PROCESSED": tmp_path,
        "require_frame": lambda frame, name, columns: None,
        "atomic_parquet": lambda frame, path: None,
    }
    exec(_function_source("_finite_numeric_values"), namespace)
    exec(_function_source("_safe_median"), namespace)
    exec(_function_source("_safe_quantile"), namespace)
    exec(_function_source("build_climatology"), namespace)

    history = pd.DataFrame({
        "station_id": ["A", "A", "B", "B"],
        "time": pd.to_datetime(
            ["2025-01-01 00:00", "2025-01-01 01:00",
             "2025-01-01 00:00", "2025-01-01 01:00"],
            utc=True,
        ),
        "tide_obs_m": [np.nan, np.nan, np.nan, np.nan],
        "surge_residual_m": [np.nan, np.nan, np.nan, np.nan],
        "WVHT": [1.0, 1.4, np.nan, np.nan],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        climatology = namespace["build_climatology"](history)

    assert climatology.attrs["available_features"] == ["WVHT"]
    assert set(climatology.attrs["dropped_features"]) == {
        "tide_obs_m", "surge_residual_m"
    }
    assert "WVHT_median" in climatology
    assert "tide_obs_m_median" not in climatology
    assert climatology.loc[climatology["station_id"] == "B", "WVHT_median"].isna().all()


def test_climatology_all_missing_uses_key_only_fallback(tmp_path) -> None:
    import warnings

    import numpy as np
    import pandas as pd

    namespace = {
        "np": np,
        "pd": pd,
        "CLIMATOLOGY_FEATURES": ["tide_obs_m", "surge_residual_m"],
        "PROCESSED": tmp_path,
        "require_frame": lambda frame, name, columns: None,
        "atomic_parquet": lambda frame, path: None,
    }
    exec(_function_source("_finite_numeric_values"), namespace)
    exec(_function_source("_safe_median"), namespace)
    exec(_function_source("_safe_quantile"), namespace)
    exec(_function_source("build_climatology"), namespace)

    history = pd.DataFrame({
        "station_id": ["A", "A"],
        "time": pd.to_datetime(
            ["2025-01-01 00:00", "2025-01-01 01:00"], utc=True
        ),
        "tide_obs_m": [np.nan, np.nan],
        "surge_residual_m": [np.nan, np.nan],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        climatology = namespace["build_climatology"](history)

    assert list(climatology.columns) == ["station_id", "month", "hour"]
    assert climatology.attrs["available_features"] == []
    assert set(climatology.attrs["dropped_features"]) == {
        "tide_obs_m", "surge_residual_m"
    }



def test_finite_median_imputer_preserves_all_missing_columns_without_warnings() -> None:
    import warnings

    import numpy as np
    import pandas as pd

    namespace = {
        "np": np,
        "pd": pd,
        "BaseEstimator": __import__("sklearn.base", fromlist=["BaseEstimator"]).BaseEstimator,
        "TransformerMixin": __import__("sklearn.base", fromlist=["TransformerMixin"]).TransformerMixin,
    }
    exec(_class_source("FiniteMedianImputer"), namespace)
    imputer = namespace["FiniteMedianImputer"](fallback_value=0.0, add_indicator=True)
    frame = pd.DataFrame({
        "available": [1.0, np.nan, 3.0],
        "unavailable": [np.nan, np.nan, np.nan],
    })

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        transformed = imputer.fit_transform(frame)
        repeated = imputer.transform(frame.iloc[[0, 1]])

    assert transformed.shape[0] == 3
    assert repeated.shape[1] == transformed.shape[1]
    assert np.isfinite(transformed).all()
    assert imputer.statistics_.tolist() == [2.0, 0.0]


def test_notebook_is_compact_and_has_no_widget_state() -> None:
    notebook = _notebook()
    assert NOTEBOOK.stat().st_size < 160_000
    assert "widgets" not in notebook.metadata
    for cell in notebook.cells:
        assert "outputId" not in cell.metadata
        colab = cell.metadata.get("colab", {})
        assert "base_uri" not in colab


def test_install_cell_uses_isolated_builds_and_source_fallback() -> None:
    source = _code_source()
    assert '"setuptools>=61"' in source
    assert '"pybind11>=2.13"' in source
    assert '"--no-build-isolation"' in source
    assert '[sys.executable, "-m", "pip", "install", "--no-build-isolation", "--editable", str(repo_dir)]' in source
    assert 'fallback_env["PUFFERFORGE_BUILD_NATIVE"] = "0"' in source
    assert 'source_dir = str(repo_dir / "python")' in source
    assert 'lines[-120:]' in source
    assert 'SURFGUARD_INSTALL_GRIB' in source
    assert 'SURFGUARD_REPO_DIR' in source
