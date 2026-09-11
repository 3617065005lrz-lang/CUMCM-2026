"""
2026 CUMCM C题问题二：滚动预测与两阶段随机线性规划（第一版）。

功能：
1. 读取附件1中的144点电价、冷启动负载先验和光伏先验；
2. 读取附件2全年小区负载与光伏实际功率并完成质量检查、宽表转长表；
3. 按问题二解读执行负载与光伏滚动预测；
4. 使用最近60个有效联合日残差，有放回抽取50个整日情景；
5. 逐日求解两阶段随机线性规划，并连续传递实际储电量；
6. 同时计算规则基线与确定性线性规划基线；
7. 输出逐时段表、每日汇总、年度对比、情景抽样、预测评价和约束校验；
8. 保存绘图缓存，所有图片均输出为PNG。

运行环境：Python 3.11+
依赖：pandas、numpy、scipy、matplotlib、openpyxl

运行示例：
    py -3.11 problem2_stochastic_programming_第一版.py --attachment1 "附件1.xlsx" --attachment2 "附件2.xlsx" --mode all
    py -3.11 problem2_stochastic_programming_第一版.py --attachment1 "附件1.xlsx" --attachment2 "附件2.xlsx" --mode solve
    py -3.11 problem2_stochastic_programming_第一版.py --output-dir "问题二随机规划输出_第一版" --mode plot

模式：
    all         数据处理、全年求解、输出表格并绘图
    preprocess  只执行数据处理、质量检查和探索性分析
    solve       数据处理、全年求解并保存绘图缓存，不生成图片
    plot        只读取绘图缓存重新生成PNG图片
"""

from __future__ import annotations

import argparse
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import coo_matrix


VERSION = "第一版"
TIME_STEP_MINUTES = 10
EXPECTED_INTERVALS = 144
EXPECTED_DAYS = 365

SCHEME_RULE = "规则基线"
SCHEME_DETERMINISTIC = "确定性线性规划"
SCHEME_STOCHASTIC = "两阶段随机线性规划"

COLOR_BLUE = "#1F3A93"
COLOR_ORANGE = "#D95F02"
COLOR_GREEN = "#1B9E77"
COLOR_RED = "#C44E52"
COLOR_PURPLE = "#6A3D9A"
COLOR_GRAY = "#667085"


@dataclass(frozen=True)
class ModelConfig:
    """问题二模型参数。"""

    interval_hours: float = 1.0 / 6.0
    charge_efficiency: float = 0.90
    discharge_efficiency: float = 0.90
    initial_storage_kwh: float = 6000.0
    minimum_storage_kwh: float = 1200.0
    maximum_storage_kwh: float = 10800.0
    maximum_power_kw: float = 5000.0
    emergency_price_multiplier: float = 5.0
    scenario_count: int = 50
    residual_window_days: int = 60
    random_seed: int = 20250911
    formal_start_date: str = "2025-02-01"
    effective_residual_start_date: str = "2025-01-15"
    numerical_tolerance: float = 1e-7
    daytime_actual_threshold_kw: float = 1.0

    @property
    def maximum_interval_energy_kwh(self) -> float:
        return self.maximum_power_kw * self.interval_hours

    def validate(self) -> None:
        if not 0 < self.interval_hours <= 24:
            raise ValueError("时间步长必须大于0且不超过24小时。")
        if not 0 < self.charge_efficiency <= 1:
            raise ValueError("充电效率必须位于(0, 1]。")
        if not 0 < self.discharge_efficiency <= 1:
            raise ValueError("放电效率必须位于(0, 1]。")
        if self.minimum_storage_kwh > self.maximum_storage_kwh:
            raise ValueError("储电量下限不能大于储电量上限。")
        if not self.minimum_storage_kwh <= self.initial_storage_kwh <= self.maximum_storage_kwh:
            raise ValueError("初始储电量必须位于储电量上下限之间。")
        if self.maximum_power_kw <= 0:
            raise ValueError("最大充放电功率必须大于0。")
        if self.emergency_price_multiplier <= 1:
            raise ValueError("紧急购电价格倍数必须大于1。")
        if self.scenario_count <= 0 or self.residual_window_days <= 0:
            raise ValueError("情景数量和残差窗口天数必须为正整数。")


def versioned_name(stem: str, extension: str) -> str:
    extension = extension if extension.startswith(".") else f".{extension}"
    return f"{stem}_{VERSION}{extension}"


def format_clock(total_minutes: int) -> str:
    if total_minutes == 1440:
        return "24:00"
    hour, minute = divmod(total_minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def interval_label(endpoint_minutes: int) -> str:
    return f"{format_clock(endpoint_minutes - TIME_STEP_MINUTES)}-{format_clock(endpoint_minutes)}"


def parse_time_endpoint(value: Any) -> float:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return float(value.hour * 60 + value.minute)
    if isinstance(value, time):
        return float(value.hour * 60 + value.minute)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return np.nan
        if 0 <= number <= 1:
            return float(round(number * 1440))
        if number.is_integer() and 0 <= number <= 1440:
            return number
        return np.nan
    text = str(value).strip().replace("：", ":").replace(" ", "")
    next_day = text.endswith("+1")
    if next_day:
        text = text[:-2]
    match = re.fullmatch(r"(\d{1,2}):(\d{1,2})(?::\d{1,2})?", text)
    if not match:
        return np.nan
    hour, minute = map(int, match.groups())
    if hour == 24 and minute == 0:
        return 1440.0
    if hour > 23 or minute > 59:
        return np.nan
    endpoint = hour * 60 + minute
    if next_day:
        endpoint += 1440
    return float(endpoint)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def read_attachment1(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到附件1：{path}")
    raw = normalize_columns(pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl"))
    required = ["时间", "电价", "小区负载", "光伏发电预测功率"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"附件1缺少字段：{missing}")
    if len(raw) != EXPECTED_INTERVALS:
        raise ValueError(f"附件1应包含{EXPECTED_INTERVALS}个时段，实际为{len(raw)}个。")
    endpoint = raw["时间"].map(parse_time_endpoint).to_numpy(dtype=float)
    if endpoint[-1] == 0:
        endpoint[-1] = 1440
    expected = np.arange(10, 1450, 10, dtype=float)
    if np.isnan(endpoint).any() or not np.array_equal(endpoint, expected):
        raise ValueError("附件1时间列不是00:10至次日00:00的连续144个10分钟时段。")
    result = pd.DataFrame(
        {
            "时段编号": np.arange(1, EXPECTED_INTERVALS + 1),
            "区间终点分钟": endpoint.astype(int),
            "时段": [interval_label(int(value)) for value in endpoint],
            "电价（元/kWh）": pd.to_numeric(raw["电价"], errors="coerce"),
            "负载先验功率（kW）": pd.to_numeric(raw["小区负载"], errors="coerce"),
            "光伏先验功率（kW）": pd.to_numeric(raw["光伏发电预测功率"], errors="coerce"),
        }
    )
    if result.iloc[:, 3:].isna().any().any():
        raise ValueError("附件1存在无法转换为数值的电价、负载或光伏数据。")
    if (result[["电价（元/kWh）", "负载先验功率（kW）", "光伏先验功率（kW）"]] < 0).any().any():
        raise ValueError("附件1存在负数，无法直接进入问题二模型。")
    return result


def _read_attachment2_sheet(path: Path, sheet_name: str) -> tuple[pd.DatetimeIndex, np.ndarray, list[Any]]:
    raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    if raw.shape[1] != EXPECTED_INTERVALS + 1:
        raise ValueError(f"工作表“{sheet_name}”应包含1列日期和144列时段。")
    dates = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0], errors="coerce")).normalize()
    values = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return dates, values, list(raw.columns[1:])


def _parse_attachment2_headers(headers: Iterable[Any]) -> np.ndarray:
    endpoint = np.array([parse_time_endpoint(value) for value in headers], dtype=float)
    if len(endpoint) and endpoint[-1] == 0:
        endpoint[-1] = 1440
    return endpoint


def read_attachment2(path: Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"找不到附件2：{path}")
    load_dates, load_kw, load_headers = _read_attachment2_sheet(path, "小区负载")
    pv_dates, pv_kw, pv_headers = _read_attachment2_sheet(path, "光伏发电实际功率")
    load_endpoint = _parse_attachment2_headers(load_headers)
    pv_endpoint = _parse_attachment2_headers(pv_headers)
    expected_endpoint = np.arange(10, 1450, 10, dtype=float)
    if np.isnan(load_endpoint).any() or not np.array_equal(load_endpoint, expected_endpoint):
        raise ValueError("附件2小区负载工作表的时段不是连续144个10分钟时段。")
    if not np.array_equal(load_endpoint, pv_endpoint):
        raise ValueError("附件2两张工作表的时段列不一致。")
    if not load_dates.equals(pv_dates):
        raise ValueError("附件2两张工作表的日期不一致。")
    return load_dates, load_kw, pv_kw, load_endpoint.astype(int)


def build_quality_tables(
    dates: pd.DatetimeIndex,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    endpoint: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_dates = pd.date_range(dates.min(), dates.max(), freq="D")
    quality_rows: list[list[Any]] = []

    def add(item: str, value: Any, requirement: str, passed: bool) -> None:
        quality_rows.append([item, value, requirement, "通过" if passed else "需复核"])

    add("日期行数", len(dates), f"应为{EXPECTED_DAYS}", len(dates) == EXPECTED_DAYS)
    add("日期范围", f"{dates.min().date()}至{dates.max().date()}", "覆盖2025年全年", dates.min() == pd.Timestamp("2025-01-01") and dates.max() == pd.Timestamp("2025-12-31"))
    add("缺失日期数", len(expected_dates.difference(dates)), "应为0", len(expected_dates.difference(dates)) == 0)
    add("重复日期数", int(dates.duplicated().sum()), "应为0", not dates.duplicated().any())
    add("每日时段数", load_kw.shape[1], f"应为{EXPECTED_INTERVALS}", load_kw.shape[1] == EXPECTED_INTERVALS)
    add("时间连续性", bool(np.array_equal(endpoint, np.arange(10, 1450, 10))), "每10分钟连续", bool(np.array_equal(endpoint, np.arange(10, 1450, 10))))
    add("负载缺失值数", int(np.isnan(load_kw).sum()), "应为0", not np.isnan(load_kw).any())
    add("光伏缺失值数", int(np.isnan(pv_kw).sum()), "应为0", not np.isnan(pv_kw).any())
    add("负载负值数", int((load_kw < 0).sum()), "应为0", not (load_kw < 0).any())
    add("光伏负值数", int((pv_kw < 0).sum()), "应为0", not (pv_kw < 0).any())
    add("负载零值数", int((load_kw == 0).sum()), "记录但不直接删除", True)
    add("光伏零值数", int((pv_kw == 0).sum()), "夜间零值视为有效数据", True)
    add("负载最小值（kW）", float(np.nanmin(load_kw)), "应为非负有限数", np.isfinite(load_kw).all() and np.nanmin(load_kw) >= 0)
    add("负载最大值（kW）", float(np.nanmax(load_kw)), "应为非负有限数", np.isfinite(load_kw).all())
    add("光伏最小值（kW）", float(np.nanmin(pv_kw)), "应为非负有限数", np.isfinite(pv_kw).all() and np.nanmin(pv_kw) >= 0)
    add("光伏最大值（kW）", float(np.nanmax(pv_kw)), "应为非负有限数", np.isfinite(pv_kw).all())
    quality = pd.DataFrame(quality_rows, columns=["检查项目", "检查结果", "判断标准", "结论"])

    anomaly_rows: list[list[Any]] = []
    for variable_name, matrix in [("小区负载", load_kw), ("光伏发电实际功率", pv_kw)]:
        for period in range(EXPECTED_INTERVALS):
            series = matrix[:, period]
            finite = series[np.isfinite(series)]
            if len(finite) == 0:
                continue
            q1, q3 = np.quantile(finite, [0.25, 0.75])
            iqr = q3 - q1
            lower = max(0.0, q1 - 3.0 * iqr)
            upper = q3 + 3.0 * iqr
            flagged = np.where((series < lower) | (series > upper) | ~np.isfinite(series))[0]
            for day_index in flagged:
                anomaly_rows.append(
                    [
                        dates[day_index].date().isoformat(),
                        period + 1,
                        interval_label(int(endpoint[period])),
                        variable_name,
                        float(series[day_index]) if np.isfinite(series[day_index]) else np.nan,
                        float(lower),
                        float(upper),
                        "按同一时段3倍四分位距标记，仅供复核，原值保留",
                    ]
                )
    anomalies = pd.DataFrame(
        anomaly_rows,
        columns=["日期", "时段编号", "时段", "变量", "原始值（kW）", "参考下限（kW）", "参考上限（kW）", "处理方式"],
    )
    before_after = pd.DataFrame(
        [
            ["日期数", len(dates), len(dates), 0],
            ["每日时段数", load_kw.shape[1], load_kw.shape[1], 0],
            ["负载数据点数", load_kw.size, load_kw.size, 0],
            ["光伏数据点数", pv_kw.size, pv_kw.size, 0],
            ["负载缺失值数", int(np.isnan(load_kw).sum()), int(np.isnan(load_kw).sum()), 0],
            ["光伏缺失值数", int(np.isnan(pv_kw).sum()), int(np.isnan(pv_kw).sum()), 0],
            ["异常标记数", 0, len(anomalies), len(anomalies)],
        ],
        columns=["项目", "预处理前", "预处理后", "变化量"],
    )
    return quality, anomalies, before_after


def build_clean_long_table(
    dates: pd.DatetimeIndex,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    endpoint: np.ndarray,
) -> pd.DataFrame:
    days = len(dates)
    repeated_dates = np.repeat(dates.to_numpy(), EXPECTED_INTERVALS)
    period_number = np.tile(np.arange(1, EXPECTED_INTERVALS + 1), days)
    endpoint_long = np.tile(endpoint, days)
    result = pd.DataFrame(
        {
            "日期": repeated_dates,
            "时段编号": period_number,
            "区间终点分钟": endpoint_long,
            "时段": np.tile([interval_label(int(value)) for value in endpoint], days),
            "星期": pd.DatetimeIndex(repeated_dates).dayofweek + 1,
            "月份": pd.DatetimeIndex(repeated_dates).month,
            "小区负载功率（kW）": load_kw.reshape(-1),
            "光伏实际功率（kW）": pv_kw.reshape(-1),
        }
    )
    result["小区耗电量（kWh）"] = result["小区负载功率（kW）"] / 6.0
    result["光伏发电量（kWh）"] = result["光伏实际功率（kW）"] / 6.0
    result["实际净需求电量（kWh）"] = result["小区耗电量（kWh）"] - result["光伏发电量（kWh）"]
    return result


def forecast_one_day(
    day_index: int,
    actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray,
    reference_load_kw: np.ndarray,
    reference_pv_kw: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    if day_index < 7:
        load_forecast = reference_load_kw.copy()
        load_source = "附件1负载先验"
    elif day_index < 14:
        load_forecast = 0.5 * actual_load_kw[day_index - 7] + 0.5 * reference_load_kw
        load_source = "附件2前7日实际负载与附件1先验等权平均"
    else:
        load_forecast = 0.5 * actual_load_kw[day_index - 7] + 0.5 * actual_load_kw[day_index - 14]
        load_source = "附件2前7日与前14日实际负载等权平均"

    pv_components: list[np.ndarray] = []
    real_history_count = 0
    for lag in (1, 2, 3):
        if day_index - lag >= 0:
            pv_components.append(actual_pv_kw[day_index - lag])
            real_history_count += 1
        else:
            pv_components.append(reference_pv_kw)
    pv_forecast = np.mean(np.vstack(pv_components), axis=0)
    pv_forecast = np.maximum(pv_forecast, 0.0)
    night_mask = pv_forecast <= tolerance
    pv_forecast[night_mask] = 0.0
    if real_history_count == 3:
        pv_source = "附件2前三日实际光伏平均"
    elif real_history_count == 0:
        pv_source = "附件1光伏先验"
    else:
        pv_source = f"附件2前{real_history_count}日实际光伏与附件1先验补足三日平均"
    return load_forecast, pv_forecast, night_mask, load_source, pv_source


def generate_scenarios(
    load_forecast_kw: np.ndarray,
    pv_forecast_kw: np.ndarray,
    night_mask: np.ndarray,
    residual_pool: list[dict[str, Any]],
    config: ModelConfig,
    day_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not residual_pool:
        return load_forecast_kw[None, :], pv_forecast_kw[None, :], []
    available = residual_pool[-config.residual_window_days :]
    rng = np.random.default_rng(day_seed)
    sampled_indices = rng.integers(0, len(available), size=config.scenario_count)
    sampled = [available[index] for index in sampled_indices]
    load_scenarios = np.vstack([load_forecast_kw + record["负载残差"] for record in sampled])
    pv_scenarios = np.vstack([pv_forecast_kw + record["光伏残差"] for record in sampled])
    load_scenarios = np.maximum(load_scenarios, 0.0)
    pv_scenarios = np.maximum(pv_scenarios, 0.0)
    pv_scenarios[:, night_mask] = 0.0
    sampled_dates = [str(record["日期"]) for record in sampled]
    return load_scenarios, pv_scenarios, sampled_dates


def _scenario_variable_offsets(scenario_index: int, periods: int) -> tuple[int, int, int, int, int]:
    base = periods + scenario_index * 5 * periods
    return base, base + periods, base + 2 * periods, base + 3 * periods, base + 4 * periods


def solve_two_stage_linear_program(
    price: np.ndarray,
    load_scenarios_kw: np.ndarray,
    pv_scenarios_kw: np.ndarray,
    initial_storage_kwh: float,
    terminal_value: float,
    config: ModelConfig,
) -> tuple[np.ndarray, dict[str, Any], OptimizeResult]:
    periods = len(price)
    scenarios = load_scenarios_kw.shape[0]
    if load_scenarios_kw.shape != pv_scenarios_kw.shape or load_scenarios_kw.shape[1] != periods:
        raise ValueError("情景矩阵与电价时段维度不一致。")
    variable_count = periods + scenarios * 5 * periods
    row_count = scenarios * 2 * periods
    objective = np.zeros(variable_count, dtype=float)
    objective[:periods] = price

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    rhs = np.zeros(row_count, dtype=float)
    probability = 1.0 / scenarios

    for scenario in range(scenarios):
        charge_offset, discharge_offset, storage_offset, waste_offset, emergency_offset = _scenario_variable_offsets(scenario, periods)
        objective[emergency_offset : emergency_offset + periods] = probability * config.emergency_price_multiplier * price
        objective[storage_offset + periods - 1] = -probability * terminal_value
        for period in range(periods):
            energy_row = scenario * 2 * periods + period
            rows.extend([energy_row] * 5)
            columns.extend(
                [period, charge_offset + period, discharge_offset + period, waste_offset + period, emergency_offset + period]
            )
            values.extend([1.0, -1.0, 1.0, -1.0, 1.0])
            rhs[energy_row] = (
                load_scenarios_kw[scenario, period] - pv_scenarios_kw[scenario, period]
            ) * config.interval_hours

            storage_row = scenario * 2 * periods + periods + period
            rows.extend([storage_row, storage_row, storage_row])
            columns.extend([charge_offset + period, discharge_offset + period, storage_offset + period])
            values.extend([-config.charge_efficiency, 1.0 / config.discharge_efficiency, 1.0])
            if period == 0:
                rhs[storage_row] = initial_storage_kwh
            else:
                rows.append(storage_row)
                columns.append(storage_offset + period - 1)
                values.append(-1.0)

    equality_matrix = coo_matrix((values, (rows, columns)), shape=(row_count, variable_count)).tocsr()
    maximum_interval_energy = config.maximum_interval_energy_kwh
    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * periods
    for _ in range(scenarios):
        bounds.extend([(0.0, maximum_interval_energy)] * periods)
        bounds.extend([(0.0, maximum_interval_energy)] * periods)
        bounds.extend([(config.minimum_storage_kwh, config.maximum_storage_kwh)] * periods)
        bounds.extend([(0.0, None)] * periods)
        bounds.extend([(0.0, None)] * periods)

    solver_result = linprog(
        c=objective,
        A_eq=equality_matrix,
        b_eq=rhs,
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not solver_result.success:
        raise RuntimeError(
            "两阶段线性规划未获得最优解。"
            f"状态码：{solver_result.status}；信息：{solver_result.message}"
        )
    solution = np.asarray(solver_result.x, dtype=float)
    solution[np.abs(solution) < config.numerical_tolerance] = 0.0
    purchase = solution[:periods]
    expected_emergency = np.zeros(periods)
    expected_end_storage = 0.0
    for scenario in range(scenarios):
        _, _, storage_offset, _, emergency_offset = _scenario_variable_offsets(scenario, periods)
        expected_emergency += probability * solution[emergency_offset : emergency_offset + periods]
        expected_end_storage += probability * solution[storage_offset + periods - 1]
    maximum_equality_residual = float(np.max(np.abs(equality_matrix @ solution - rhs)))
    objective_with_constant = float(solver_result.fun + terminal_value * initial_storage_kwh)
    diagnostics = {
        "情景数": scenarios,
        "目标函数值（元）": objective_with_constant,
        "预期应急购电量（kWh）": float(expected_emergency.sum()),
        "预期日末储电量（kWh）": float(expected_end_storage),
        "最大模型等式残差（kWh）": maximum_equality_residual,
        "求解状态": "最优解",
        "求解器状态码": int(solver_result.status),
        "求解器信息": str(solver_result.message),
    }
    return purchase, diagnostics, solver_result


def build_rule_purchase(
    load_scenarios_kw: np.ndarray,
    pv_scenarios_kw: np.ndarray,
    config: ModelConfig,
) -> np.ndarray:
    net_energy = (load_scenarios_kw - pv_scenarios_kw) * config.interval_hours
    if len(net_energy) == 1:
        purchase = net_energy[0]
    else:
        purchase = np.quantile(net_energy, 0.80, axis=0, method="linear")
    return np.maximum(purchase, 0.0)


def simulate_actual_operation(
    planned_purchase_kwh: np.ndarray,
    actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray,
    initial_storage_kwh: float,
    price: np.ndarray,
    config: ModelConfig,
) -> dict[str, np.ndarray]:
    periods = len(planned_purchase_kwh)
    charge = np.zeros(periods)
    discharge = np.zeros(periods)
    emergency = np.zeros(periods)
    waste = np.zeros(periods)
    storage_start = np.zeros(periods)
    storage_end = np.zeros(periods)
    storage = float(initial_storage_kwh)
    maximum_interval_energy = config.maximum_interval_energy_kwh
    load_energy = actual_load_kw * config.interval_hours
    pv_energy = actual_pv_kw * config.interval_hours

    for period in range(periods):
        storage_start[period] = storage
        deviation = planned_purchase_kwh[period] + pv_energy[period] - load_energy[period]
        if deviation > 0:
            charge[period] = min(
                deviation,
                maximum_interval_energy,
                max(0.0, (config.maximum_storage_kwh - storage) / config.charge_efficiency),
            )
            waste[period] = max(0.0, deviation - charge[period])
            storage += config.charge_efficiency * charge[period]
        elif deviation < 0:
            shortage = -deviation
            discharge[period] = min(
                shortage,
                maximum_interval_energy,
                max(0.0, config.discharge_efficiency * (storage - config.minimum_storage_kwh)),
            )
            emergency[period] = max(0.0, shortage - discharge[period])
            storage -= discharge[period] / config.discharge_efficiency
        storage = min(config.maximum_storage_kwh, max(config.minimum_storage_kwh, storage))
        storage_end[period] = storage

    plan_cost = price * planned_purchase_kwh
    emergency_cost = config.emergency_price_multiplier * price * emergency
    balance_residual = planned_purchase_kwh + pv_energy + discharge + emergency - load_energy - charge - waste
    storage_residual = storage_end - storage_start - config.charge_efficiency * charge + discharge / config.discharge_efficiency
    return {
        "负载电量（kWh）": load_energy,
        "光伏发电量（kWh）": pv_energy,
        "计划购电量（kWh）": planned_purchase_kwh,
        "实际充电量（kWh）": charge,
        "实际放电量（kWh）": discharge,
        "实际紧急购电量（kWh）": emergency,
        "弃光或未利用电量（kWh）": waste,
        "时段初储电量（kWh）": storage_start,
        "时段末储电量（kWh）": storage_end,
        "计划购电成本（元）": plan_cost,
        "紧急购电成本（元）": emergency_cost,
        "时段总购电成本（元）": plan_cost + emergency_cost,
        "实际能量平衡残差（kWh）": balance_residual,
        "实际储能递推残差（kWh）": storage_residual,
    }


def _daily_detail_frame(
    date: pd.Timestamp,
    scheme: str,
    endpoint: np.ndarray,
    price: np.ndarray,
    load_forecast_kw: np.ndarray,
    pv_forecast_kw: np.ndarray,
    actual_load_kw: np.ndarray,
    actual_pv_kw: np.ndarray,
    operation: dict[str, np.ndarray],
    load_source: str,
    pv_source: str,
    residual_pool_size: int,
    scenario_count: int,
    day_seed: int,
) -> pd.DataFrame:
    periods = len(endpoint)
    frame = pd.DataFrame(
        {
            "方案": scheme,
            "日期": np.repeat(date, periods),
            "时段编号": np.arange(1, periods + 1),
            "区间终点分钟": endpoint,
            "时段": [interval_label(int(value)) for value in endpoint],
            "电价（元/kWh）": price,
            "负载预测功率（kW）": load_forecast_kw,
            "光伏预测功率（kW）": pv_forecast_kw,
            "实际负载功率（kW）": actual_load_kw,
            "实际光伏功率（kW）": actual_pv_kw,
            "负载预测来源": np.repeat(load_source, periods),
            "光伏预测来源": np.repeat(pv_source, periods),
            "决策前有效残差数": np.repeat(residual_pool_size, periods),
            "优化情景数": np.repeat(scenario_count, periods),
            "当日随机种子": np.repeat(day_seed, periods),
        }
    )
    for column, values in operation.items():
        frame[column] = values
    frame["负载预测残差（kW）"] = frame["实际负载功率（kW）"] - frame["负载预测功率（kW）"]
    frame["光伏预测残差（kW）"] = frame["实际光伏功率（kW）"] - frame["光伏预测功率（kW）"]
    frame["预测净需求电量（kWh）"] = (
        frame["负载预测功率（kW）"] - frame["光伏预测功率（kW）"]
    ) / 6.0
    frame["实际净需求电量（kWh）"] = frame["负载电量（kWh）"] - frame["光伏发电量（kWh）"]
    return frame


def build_daily_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scheme, date), group in detail.groupby(["方案", "日期"], sort=True):
        rows.append(
            {
                "方案": scheme,
                "日期": pd.Timestamp(date),
                "日初储电量（kWh）": float(group.iloc[0]["时段初储电量（kWh）"]),
                "日末储电量（kWh）": float(group.iloc[-1]["时段末储电量（kWh）"]),
                "最低储电量（kWh）": float(group["时段末储电量（kWh）"].min()),
                "最高储电量（kWh）": float(group["时段末储电量（kWh）"].max()),
                "计划购电总量（kWh）": float(group["计划购电量（kWh）"].sum()),
                "计划购电成本（元）": float(group["计划购电成本（元）"].sum()),
                "紧急购电总量（kWh）": float(group["实际紧急购电量（kWh）"].sum()),
                "紧急购电成本（元）": float(group["紧急购电成本（元）"].sum()),
                "总购电成本（元）": float(group["时段总购电成本（元）"].sum()),
                "实际负载电量（kWh）": float(group["负载电量（kWh）"].sum()),
                "实际光伏发电量（kWh）": float(group["光伏发电量（kWh）"].sum()),
                "实际充电总量（kWh）": float(group["实际充电量（kWh）"].sum()),
                "实际放电总量（kWh）": float(group["实际放电量（kWh）"].sum()),
                "弃光或未利用总量（kWh）": float(group["弃光或未利用电量（kWh）"].sum()),
                "紧急购电时段数": int((group["实际紧急购电量（kWh）"] > 1e-7).sum()),
                "最大实际能量平衡残差（kWh）": float(group["实际能量平衡残差（kWh）"].abs().max()),
                "最大实际储能递推残差（kWh）": float(group["实际储能递推残差（kWh）"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def build_annual_comparison(daily_summary: pd.DataFrame, formal_start: pd.Timestamp) -> pd.DataFrame:
    formal = daily_summary.loc[daily_summary["日期"] >= formal_start].copy()
    rows: list[dict[str, Any]] = []
    for scheme, group in formal.groupby("方案", sort=False):
        rows.append(
            {
                "方案": scheme,
                "统计开始日期": group["日期"].min(),
                "统计结束日期": group["日期"].max(),
                "统计天数": group["日期"].nunique(),
                "计划购电总量（kWh）": group["计划购电总量（kWh）"].sum(),
                "计划购电成本（元）": group["计划购电成本（元）"].sum(),
                "紧急购电总量（kWh）": group["紧急购电总量（kWh）"].sum(),
                "紧急购电成本（元）": group["紧急购电成本（元）"].sum(),
                "全年总购电成本（元）": group["总购电成本（元）"].sum(),
                "实际负载电量（kWh）": group["实际负载电量（kWh）"].sum(),
                "实际光伏发电量（kWh）": group["实际光伏发电量（kWh）"].sum(),
                "实际充电总量（kWh）": group["实际充电总量（kWh）"].sum(),
                "实际放电总量（kWh）": group["实际放电总量（kWh）"].sum(),
                "弃光或未利用总量（kWh）": group["弃光或未利用总量（kWh）"].sum(),
                "紧急购电时段数": group["紧急购电时段数"].sum(),
                "最低储电量（kWh）": group["最低储电量（kWh）"].min(),
                "最高储电量（kWh）": group["最高储电量（kWh）"].max(),
            }
        )
    comparison = pd.DataFrame(rows)
    rule_cost_series = comparison.loc[comparison["方案"] == SCHEME_RULE, "全年总购电成本（元）"]
    deterministic_cost_series = comparison.loc[comparison["方案"] == SCHEME_DETERMINISTIC, "全年总购电成本（元）"]
    rule_cost = float(rule_cost_series.iloc[0]) if len(rule_cost_series) else np.nan
    deterministic_cost = float(deterministic_cost_series.iloc[0]) if len(deterministic_cost_series) else np.nan
    comparison["相对规则基线节省金额（元）"] = rule_cost - comparison["全年总购电成本（元）"]
    comparison["相对规则基线节省比例"] = comparison["相对规则基线节省金额（元）"] / rule_cost
    comparison["相对确定性线性规划节省金额（元）"] = deterministic_cost - comparison["全年总购电成本（元）"]
    comparison["相对确定性线性规划节省比例"] = comparison["相对确定性线性规划节省金额（元）"] / deterministic_cost
    return comparison


def _metric_rows(
    prediction_detail: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    stage_name: str,
    config: ModelConfig,
) -> list[dict[str, Any]]:
    selected = prediction_detail.loc[
        prediction_detail["日期"].between(start_date, end_date, inclusive="both")
    ].copy()
    rows: list[dict[str, Any]] = []
    for variable, actual_column, forecast_column in [
        ("小区负载", "实际负载功率（kW）", "负载预测功率（kW）"),
        ("光伏发电", "实际光伏功率（kW）", "光伏预测功率（kW）"),
    ]:
        actual = selected[actual_column].to_numpy(dtype=float)
        forecast = selected[forecast_column].to_numpy(dtype=float)
        error = actual - forecast
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        if variable == "小区负载":
            valid_ratio = actual > config.numerical_tolerance
        else:
            valid_ratio = actual > config.daytime_actual_threshold_kw
        mape = float(np.mean(np.abs(error[valid_ratio] / actual[valid_ratio]))) if valid_ratio.any() else np.nan
        mean_actual = float(np.mean(actual))
        normalized_mae = mae / mean_actual if mean_actual > config.numerical_tolerance else np.nan
        rows.append(
            {
                "评价阶段": stage_name,
                "开始日期": start_date,
                "结束日期": end_date,
                "变量": variable,
                "样本数": len(actual),
                "MAE（kW）": mae,
                "RMSE（kW）": rmse,
                "MAPE": mape,
                "归一化MAE": normalized_mae,
                "平均误差（kW）": float(np.mean(error)),
            }
        )
    return rows


def build_prediction_metrics(prediction_detail: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    start = prediction_detail["日期"].min()
    end = prediction_detail["日期"].max()
    formal_start = pd.Timestamp(config.formal_start_date)
    effective_start = pd.Timestamp(config.effective_residual_start_date)
    rows: list[dict[str, Any]] = []
    if start <= effective_start - pd.Timedelta(days=1):
        rows.extend(
            _metric_rows(
                prediction_detail,
                start,
                min(end, effective_start - pd.Timedelta(days=1)),
                "冷启动期（不纳入正式评价）",
                config,
            )
        )
    accumulation_end = formal_start - pd.Timedelta(days=1)
    if end >= effective_start:
        rows.extend(
            _metric_rows(
                prediction_detail,
                effective_start,
                min(end, accumulation_end),
                "有效残差积累期",
                config,
            )
        )
    if end >= formal_start:
        rows.extend(_metric_rows(prediction_detail, formal_start, end, "正式评价期", config))
    return pd.DataFrame(rows)


def build_validation_table(
    detail: pd.DataFrame,
    solver_log: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    tolerance = max(config.numerical_tolerance * 100, 1e-5)
    rows: list[list[Any]] = []

    def add(item: str, value: Any, requirement: str, passed: bool) -> None:
        rows.append([item, value, requirement, "通过" if passed else "需复核"])

    add("线性规划求解失败天数", int((solver_log["求解状态"] != "最优解").sum()), "应为0", bool((solver_log["求解状态"] == "最优解").all()))
    add("最大模型等式残差", float(solver_log["最大模型等式残差（kWh）"].max()), f"不超过{tolerance}", float(solver_log["最大模型等式残差（kWh）"].max()) <= tolerance)
    add("最大实际能量平衡残差", float(detail["实际能量平衡残差（kWh）"].abs().max()), f"不超过{tolerance}", float(detail["实际能量平衡残差（kWh）"].abs().max()) <= tolerance)
    add("最大实际储能递推残差", float(detail["实际储能递推残差（kWh）"].abs().max()), f"不超过{tolerance}", float(detail["实际储能递推残差（kWh）"].abs().max()) <= tolerance)
    add("最低实际储电量", float(detail["时段末储电量（kWh）"].min()), f"不低于{config.minimum_storage_kwh}", float(detail["时段末储电量（kWh）"].min()) >= config.minimum_storage_kwh - tolerance)
    add("最高实际储电量", float(detail["时段末储电量（kWh）"].max()), f"不高于{config.maximum_storage_kwh}", float(detail["时段末储电量（kWh）"].max()) <= config.maximum_storage_kwh + tolerance)
    add("最大实际充电量", float(detail["实际充电量（kWh）"].max()), f"不高于{config.maximum_interval_energy_kwh}", float(detail["实际充电量（kWh）"].max()) <= config.maximum_interval_energy_kwh + tolerance)
    add("最大实际放电量", float(detail["实际放电量（kWh）"].max()), f"不高于{config.maximum_interval_energy_kwh}", float(detail["实际放电量（kWh）"].max()) <= config.maximum_interval_energy_kwh + tolerance)
    for column in ["计划购电量（kWh）", "实际充电量（kWh）", "实际放电量（kWh）", "实际紧急购电量（kWh）", "弃光或未利用电量（kWh）"]:
        count = int((detail[column] < -tolerance).sum())
        add(f"{column}负值数", count, "应为0", count == 0)
    for scheme, group in detail.groupby("方案"):
        ordered = group.sort_values(["日期", "时段编号"])
        day_end = ordered.groupby("日期")["时段末储电量（kWh）"].last()
        next_start = ordered.groupby("日期")["时段初储电量（kWh）"].first().shift(-1)
        difference = (day_end.iloc[:-1].to_numpy() - next_start.iloc[:-1].to_numpy()) if len(day_end) > 1 else np.array([0.0])
        maximum_difference = float(np.max(np.abs(difference)))
        add(f"{scheme}跨日SOC最大衔接误差", maximum_difference, f"不超过{tolerance}", maximum_difference <= tolerance)
    return pd.DataFrame(rows, columns=["校验项目", "实际结果", "判断标准", "结论"])


def build_parameter_table(config: ModelConfig, terminal_value: float) -> pd.DataFrame:
    rows = [
        ["时间步长", config.interval_hours, "小时"],
        ["每日时段数", EXPECTED_INTERVALS, "个"],
        ["充电效率", config.charge_efficiency, "比例"],
        ["放电效率", config.discharge_efficiency, "比例"],
        ["初始储电量", config.initial_storage_kwh, "kWh"],
        ["储电量下限", config.minimum_storage_kwh, "kWh"],
        ["储电量上限", config.maximum_storage_kwh, "kWh"],
        ["最大充放电功率", config.maximum_power_kw, "kW"],
        ["单时段最大充放电量", config.maximum_interval_energy_kwh, "kWh"],
        ["紧急购电价格倍数", config.emergency_price_multiplier, "倍"],
        ["联合残差窗口", config.residual_window_days, "日"],
        ["抽样情景数", config.scenario_count, "个"],
        ["固定随机种子", config.random_seed, ""],
        ["终端价值系数", terminal_value, "元/kWh"],
        ["正式统计开始日期", config.formal_start_date, ""],
        ["有效残差开始日期", config.effective_residual_start_date, ""],
    ]
    return pd.DataFrame(rows, columns=["参数", "数值", "单位或说明"])


def run_annual_rolling_model(
    attachment1: pd.DataFrame,
    dates: pd.DatetimeIndex,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    endpoint: np.ndarray,
    config: ModelConfig,
) -> dict[str, pd.DataFrame]:
    config.validate()
    price = attachment1["电价（元/kWh）"].to_numpy(dtype=float)
    reference_load = attachment1["负载先验功率（kW）"].to_numpy(dtype=float)
    reference_pv = attachment1["光伏先验功率（kW）"].to_numpy(dtype=float)
    valley_price = float(np.min(price))
    terminal_value = valley_price / config.charge_efficiency
    effective_residual_start = pd.Timestamp(config.effective_residual_start_date)

    residual_pool: list[dict[str, Any]] = []
    storage_state = {
        SCHEME_RULE: config.initial_storage_kwh,
        SCHEME_DETERMINISTIC: config.initial_storage_kwh,
        SCHEME_STOCHASTIC: config.initial_storage_kwh,
    }
    all_detail: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    scenario_sample_rows: list[list[Any]] = []
    solver_rows: list[dict[str, Any]] = []

    for day_index, date in enumerate(dates):
        load_forecast, pv_forecast, night_mask, load_source, pv_source = forecast_one_day(
            day_index,
            load_kw,
            pv_kw,
            reference_load,
            reference_pv,
            config.numerical_tolerance,
        )
        day_seed = config.random_seed + day_index
        available_count = min(len(residual_pool), config.residual_window_days)
        load_scenarios, pv_scenarios, sampled_dates = generate_scenarios(
            load_forecast,
            pv_forecast,
            night_mask,
            residual_pool,
            config,
            day_seed,
        )
        for scenario_number, sampled_date in enumerate(sampled_dates, start=1):
            scenario_sample_rows.append(
                [date, scenario_number, sampled_date, day_seed, available_count, config.scenario_count]
            )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "日期": np.repeat(date, EXPECTED_INTERVALS),
                    "时段编号": np.arange(1, EXPECTED_INTERVALS + 1),
                    "区间终点分钟": endpoint,
                    "时段": [interval_label(int(value)) for value in endpoint],
                    "负载预测功率（kW）": load_forecast,
                    "实际负载功率（kW）": load_kw[day_index],
                    "负载预测残差（kW）": load_kw[day_index] - load_forecast,
                    "光伏预测功率（kW）": pv_forecast,
                    "实际光伏功率（kW）": pv_kw[day_index],
                    "光伏预测残差（kW）": pv_kw[day_index] - pv_forecast,
                    "是否预测夜间": np.where(night_mask, "是", "否"),
                    "负载预测来源": np.repeat(load_source, EXPECTED_INTERVALS),
                    "光伏预测来源": np.repeat(pv_source, EXPECTED_INTERVALS),
                    "是否有效联合残差日": np.repeat("是" if date >= effective_residual_start else "否", EXPECTED_INTERVALS),
                }
            )
        )

        stochastic_purchase, stochastic_diagnostics, _ = solve_two_stage_linear_program(
            price,
            load_scenarios,
            pv_scenarios,
            storage_state[SCHEME_STOCHASTIC],
            terminal_value,
            config,
        )
        deterministic_purchase, deterministic_diagnostics, _ = solve_two_stage_linear_program(
            price,
            load_forecast[None, :],
            pv_forecast[None, :],
            storage_state[SCHEME_DETERMINISTIC],
            terminal_value,
            config,
        )
        rule_purchase = build_rule_purchase(load_scenarios, pv_scenarios, config)

        purchases = {
            SCHEME_RULE: rule_purchase,
            SCHEME_DETERMINISTIC: deterministic_purchase,
            SCHEME_STOCHASTIC: stochastic_purchase,
        }
        diagnostics_by_scheme = {
            SCHEME_DETERMINISTIC: deterministic_diagnostics,
            SCHEME_STOCHASTIC: stochastic_diagnostics,
        }
        for scheme in [SCHEME_RULE, SCHEME_DETERMINISTIC, SCHEME_STOCHASTIC]:
            operation = simulate_actual_operation(
                purchases[scheme],
                load_kw[day_index],
                pv_kw[day_index],
                storage_state[scheme],
                price,
                config,
            )
            daily_frame = _daily_detail_frame(
                date,
                scheme,
                endpoint,
                price,
                load_forecast,
                pv_forecast,
                load_kw[day_index],
                pv_kw[day_index],
                operation,
                load_source,
                pv_source,
                available_count,
                load_scenarios.shape[0] if scheme != SCHEME_DETERMINISTIC else 1,
                day_seed,
            )
            all_detail.append(daily_frame)
            storage_state[scheme] = float(operation["时段末储电量（kWh）"][-1])
            if scheme in diagnostics_by_scheme:
                solver_rows.append(
                    {
                        "日期": date,
                        "方案": scheme,
                        **diagnostics_by_scheme[scheme],
                    }
                )

        if date >= effective_residual_start:
            residual_pool.append(
                {
                    "日期": date.date().isoformat(),
                    "负载残差": load_kw[day_index] - load_forecast,
                    "光伏残差": pv_kw[day_index] - pv_forecast,
                }
            )

    detail = pd.concat(all_detail, ignore_index=True)
    prediction_detail = pd.concat(prediction_frames, ignore_index=True)
    scenario_samples = pd.DataFrame(
        scenario_sample_rows,
        columns=["决策日期", "情景编号", "抽中残差日期", "当日随机种子", "可用有效残差数", "目标情景数"],
    )
    solver_log = pd.DataFrame(solver_rows)
    daily_summary = build_daily_summary(detail)
    formal_start = pd.Timestamp(config.formal_start_date)
    annual_comparison = build_annual_comparison(daily_summary, formal_start)
    prediction_metrics = build_prediction_metrics(prediction_detail, config)
    validation = build_validation_table(detail, solver_log, config)
    parameters = build_parameter_table(config, terminal_value)
    return {
        "逐时段结果": detail,
        "预测结果": prediction_detail,
        "情景抽样记录": scenario_samples,
        "求解日志": solver_log,
        "每日汇总": daily_summary,
        "年度对比": annual_comparison,
        "预测指标": prediction_metrics,
        "约束校验": validation,
        "模型参数": parameters,
    }


def prepare_output_directories(output_dir: Path) -> tuple[Path, Path, Path]:
    table_dir = output_dir / f"数据表_{VERSION}"
    cache_dir = output_dir / f"绘图缓存_{VERSION}"
    image_dir = output_dir / f"图片_{VERSION}"
    for directory in (table_dir, cache_dir, image_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return table_dir, cache_dir, image_dir


def _fallback_path(path: Path) -> Path:
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_未覆盖_{stamp}{path.suffix}")


def write_csv(data: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}_临时_{uuid.uuid4().hex}.csv")
    data.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.6f")
    try:
        os.replace(temporary, path)
        return path
    except PermissionError:
        fallback = _fallback_path(path)
        os.replace(temporary, fallback)
        return fallback


def _style_excel(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    workbook = load_workbook(path)
    fill = PatternFill("solid", fgColor="1F4E78")
    font = Font(color="FFFFFF", bold=True)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = fill
            cell.font = font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for column_cells in worksheet.iter_cols(min_row=1, max_row=min(worksheet.max_row, 300)):
            width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(width + 2, 10), 28)
    workbook.save(path)


def write_excel(sheets: dict[str, pd.DataFrame], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}_临时_{uuid.uuid4().hex}.xlsx")
    with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            data.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    _style_excel(temporary)
    try:
        os.replace(temporary, path)
        return path
    except PermissionError:
        fallback = _fallback_path(path)
        os.replace(temporary, fallback)
        return fallback


def configure_chinese_plot_style() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#999999",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "grid.color": "#D8DEE8",
            "grid.linewidth": 0.5,
            "grid.linestyle": "--",
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _style_axes(axis: plt.Axes, grid_axis: str = "y") -> None:
    """应用统一的论文折线图与统计图坐标样式。"""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#999999")
    axis.spines["bottom"].set_color("#999999")
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(colors="#333333", direction="out", length=4)
    axis.grid(axis=grid_axis, linestyle="--", linewidth=0.5, color="#CCCCCC", alpha=0.5)
    axis.set_axisbelow(True)

def _format_scientific_y(value: float, _position: int) -> str:
    """把 1e7 这类科学计数法显示成 1 × 10^7 形式。"""
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for exponent in (9, 8, 7, 6, 5, 4, 3):
        scale = 10.0 ** exponent
        if magnitude >= scale:
            mantissa = magnitude / scale
            if abs(mantissa - round(mantissa)) < 1e-9:
                mantissa_text = f"{int(round(mantissa))}"
            else:
                mantissa_text = f"{mantissa:.1f}"
            return f"{sign}${mantissa_text} \\times 10^{{{exponent}}}$"
    return f"{sign}{magnitude:.0f}"


def _format_year_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axis.tick_params(axis="x", rotation=0)


def _set_time_axis(axis: plt.Axes) -> None:
    ticks = np.arange(0, 145, 12)
    labels = [f"{hour}:00" for hour in range(0, 25, 2)]
    axis.set_xlim(1, 144)
    axis.set_xticks(np.clip(ticks, 1, 144))
    axis.set_xticklabels(labels)
    axis.set_xlabel("时刻")


def _save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, format="png")
    plt.close(figure)


def plot_annual_energy(clean_long: pd.DataFrame, image_dir: Path, dpi: int) -> None:
    daily = clean_long.groupby("日期", as_index=False).agg(
        小区耗电量=("小区耗电量（kWh）", "sum"),
        光伏发电量=("光伏发电量（kWh）", "sum"),
    )
    figure, axis = plt.subplots(figsize=(14, 6.5))
    axis.plot(daily["日期"], daily["小区耗电量"], color=COLOR_BLUE, linewidth=1.8, label="小区耗电量")
    axis.plot(daily["日期"], daily["光伏发电量"], color=COLOR_ORANGE, linewidth=1.8, linestyle="-", label="光伏发电量")
    axis.set_xlabel("日期")
    axis.set_ylabel("日电量（kWh）")
    axis.set_ylim(bottom=0)
    _style_axes(axis)
    _format_year_axis(axis)
    axis.legend(loc="upper right", ncol=2)
    _save_figure(figure, image_dir / versioned_name("图1_全年每日负载与光伏发电量", ".png"), dpi)


def build_daily_mean_power(clean_long: pd.DataFrame) -> pd.DataFrame:
    """计算每天144个时段的平均负载功率和平均光伏功率。"""
    return clean_long.groupby("日期", as_index=False).agg(
        **{
            "日均负载功率（kW）": ("小区负载功率（kW）", "mean"),
            "日均光伏功率（kW）": ("光伏实际功率（kW）", "mean"),
        }
    )


def plot_daily_mean_boxplot(clean_long: pd.DataFrame, image_dir: Path, dpi: int) -> None:
    daily_mean = build_daily_mean_power(clean_long)
    figure, axis = plt.subplots(figsize=(14, 6.5))
    boxplot = axis.boxplot(
        [daily_mean["日均负载功率（kW）"], daily_mean["日均光伏功率（kW）"]],
        tick_labels=["日均负载", "日均光伏"],
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        medianprops={"color": "#000000", "linewidth": 2.0},
        meanprops={"marker": "D", "markerfacecolor": COLOR_RED, "markeredgecolor": "white", "markersize": 6},
        whiskerprops={"color": "#555555", "linewidth": 1.0},
        capprops={"color": "#555555", "linewidth": 1.0},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "#999999", "markeredgecolor": "#666666", "alpha": 0.55},
    )
    for patch, color in zip(boxplot["boxes"], [COLOR_BLUE, COLOR_ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.8)
    axis.set_ylabel("功率（kW）")
    axis.set_ylim(bottom=0)
    _style_axes(axis)
    axis.text(0.99, 0.02, "箱体：四分位距　中线：中位数　菱形：均值", transform=axis.transAxes, ha="right", va="bottom", fontsize=9, color=COLOR_GRAY)
    _save_figure(figure, image_dir / versioned_name("图2_全年日均负载与光伏箱线图", ".png"), dpi)


def plot_annual_prediction_metrics(
    prediction: pd.DataFrame,
    image_dir: Path,
    dpi: int,
    config: ModelConfig,
) -> None:
    metrics = build_prediction_metrics(prediction, config)
    formal = metrics.loc[metrics["评价阶段"].eq("正式评价期")].set_index("变量")
    variables = ["小区负载", "光伏发电"]
    x = np.arange(len(variables), dtype=float)
    width = 0.32
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    mae = formal.reindex(variables)["MAE（kW）"].to_numpy(dtype=float)
    rmse = formal.reindex(variables)["RMSE（kW）"].to_numpy(dtype=float)
    bars_mae = axes[0].bar(x - width / 2, mae, width, color=COLOR_BLUE, edgecolor="#333333", linewidth=0.6, label="MAE")
    bars_rmse = axes[0].bar(x + width / 2, rmse, width, color=COLOR_ORANGE, edgecolor="#333333", linewidth=0.6, label="RMSE")
    axes[0].set_xticks(x, variables)
    axes[0].set_ylabel("预测误差（kW）")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", ncol=2)
    axes[0].bar_label(bars_mae, fmt="%.1f", padding=3, fontsize=8)
    axes[0].bar_label(bars_rmse, fmt="%.1f", padding=3, fontsize=8)
    _style_axes(axes[0])

    mape = formal.reindex(variables)["MAPE"].to_numpy(dtype=float) * 100.0
    normalized_mae = formal.reindex(variables)["归一化MAE"].to_numpy(dtype=float) * 100.0
    bars_mape = axes[1].bar(x - width / 2, mape, width, color=COLOR_GREEN, edgecolor="#333333", linewidth=0.6, label="MAPE")
    bars_nmae = axes[1].bar(x + width / 2, normalized_mae, width, color=COLOR_PURPLE, edgecolor="#333333", linewidth=0.6, label="归一化MAE")
    axes[1].set_xticks(x, variables)
    axes[1].set_ylabel("相对误差（%）")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].bar_label(bars_mape, fmt="%.2f%%", padding=3, fontsize=8)
    axes[1].bar_label(bars_nmae, fmt="%.2f%%", padding=3, fontsize=8)
    _style_axes(axes[1])

    figure.tight_layout(w_pad=3.0)
    _save_figure(figure, image_dir / versioned_name("图3_全年预测误差指标", ".png"), dpi)


def plot_representative_prediction_days(
    prediction: pd.DataFrame,
    image_dir: Path,
    dpi: int,
) -> None:
    representative_dates = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
    figure, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    shared_handles: list[Any] = []
    shared_labels: list[str] = []
    for axis, date_text in zip(axes.flat, representative_dates):
        date = pd.Timestamp(date_text)
        selected = prediction.loc[pd.to_datetime(prediction["日期"]).eq(date)].sort_values("时段编号")
        if len(selected) != EXPECTED_INTERVALS:
            raise ValueError(f"预测缓存中缺少代表日期{date_text}的完整144个时段。")
        x = selected["时段编号"].to_numpy(dtype=float)
        handles = [
            axis.plot(x, selected["实际负载功率（kW）"], color=COLOR_BLUE, linewidth=1.8, linestyle="-", label="实际负载")[0],
            axis.plot(x, selected["负载预测功率（kW）"], color="#6C8CD5", linewidth=1.6, linestyle="--", label="负载预测")[0],
            axis.plot(x, selected["实际光伏功率（kW）"], color=COLOR_ORANGE, linewidth=1.8, linestyle="-", label="实际光伏")[0],
            axis.plot(x, selected["光伏预测功率（kW）"], color="#F3A55B", linewidth=1.6, linestyle="--", label="光伏预测")[0],
        ]
        if not shared_handles:
            shared_handles = handles
            shared_labels = [handle.get_label() for handle in handles]
        axis.text(0.02, 0.96, date_text, transform=axis.transAxes, ha="left", va="top", fontsize=10, color="#333333")
        axis.set_ylim(bottom=0)
        axis.set_ylabel("功率（kW）")
        _set_time_axis(axis)
        _style_axes(axis)
    figure.legend(shared_handles, shared_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4)
    figure.tight_layout(rect=(0, 0.06, 1, 1), h_pad=2.0, w_pad=2.0)
    _save_figure(figure, image_dir / versioned_name("图4_四个代表日期预测曲线", ".png"), dpi)


def build_annual_purchase_series(detail: pd.DataFrame) -> pd.DataFrame:
    """将主模型逐时段购电与净需求汇总为逐日全年序列。"""
    main = detail.loc[detail["方案"].eq(SCHEME_STOCHASTIC)].copy()
    return main.groupby("日期", as_index=False).agg(
        **{
            "计划购电量（kWh）": ("计划购电量（kWh）", "sum"),
            "实际净需求电量（kWh）": ("实际净需求电量（kWh）", "sum"),
            "实际紧急购电量（kWh）": ("实际紧急购电量（kWh）", "sum"),
        }
    )


def build_monthly_mean_purchase_series(detail: pd.DataFrame) -> pd.DataFrame:
    """先计算每日总量，再计算各月的日均计划购电、净需求和紧急购电量。"""
    daily = build_annual_purchase_series(detail)
    daily["月份"] = pd.to_datetime(daily["日期"]).dt.month
    return daily.groupby("月份", as_index=False).agg(
        **{
            "月均计划购电量（kWh/日）": ("计划购电量（kWh）", "mean"),
            "月均实际净需求电量（kWh/日）": ("实际净需求电量（kWh）", "mean"),
            "月均紧急购电量（kWh/日）": ("实际紧急购电量（kWh）", "mean"),
        }
    )


def plot_monthly_mean_purchase_and_net_demand(
    detail: pd.DataFrame,
    image_dir: Path,
    dpi: int,
) -> None:
    monthly = build_monthly_mean_purchase_series(detail)
    x = np.arange(len(monthly), dtype=float)
    width = 0.25
    figure, axis = plt.subplots(figsize=(14, 6.5))
    axis.bar(
        x - width,
        monthly["月均计划购电量（kWh/日）"],
        width,
        color=COLOR_BLUE,
        edgecolor="#333333",
        linewidth=0.6,
        label="月均计划购电量",
    )
    axis.bar(
        x,
        monthly["月均实际净需求电量（kWh/日）"],
        width,
        color=COLOR_ORANGE,
        edgecolor="#333333",
        linewidth=0.6,
        label="月均实际净需求电量",
    )
    axis.bar(
        x + width,
        monthly["月均紧急购电量（kWh/日）"],
        width,
        color=COLOR_RED,
        edgecolor="#333333",
        linewidth=0.6,
        label="月均紧急购电量",
    )
    axis.set_xticks(x, [f"{month}月" for month in monthly["月份"]])
    axis.set_xlabel("月份")
    axis.set_ylabel("月均日电量（kWh/日）")
    axis.set_ylim(bottom=0)
    _style_axes(axis)
    axis.legend(loc="upper right", bbox_to_anchor=(1.0, 1.05), ncol=3)
    _save_figure(figure, image_dir / versioned_name("图5_月均计划购电实际净需求与紧急购电量", ".png"), dpi)


def plot_cumulative_cost(daily: pd.DataFrame, image_dir: Path, dpi: int, formal_start: str) -> None:
    selected = daily.loc[pd.to_datetime(daily["日期"]) >= pd.Timestamp(formal_start)].copy()
    figure, axis = plt.subplots(figsize=(14, 6.5))
    styles = {
        SCHEME_RULE: (COLOR_GRAY, "-"),
        SCHEME_DETERMINISTIC: (COLOR_ORANGE, "-"),
        SCHEME_STOCHASTIC: (COLOR_BLUE, "-"),
    }
    end_date = pd.Timestamp(selected["日期"].max())
    for scheme in [SCHEME_RULE, SCHEME_DETERMINISTIC, SCHEME_STOCHASTIC]:
        group = selected.loc[selected["方案"].eq(scheme)].sort_values("日期")
        color, line_style = styles[scheme]
        cumulative = group["总购电成本（元）"].cumsum()
        axis.plot(
            group["日期"],
            cumulative,
            color=color,
            linewidth=2.0,
            linestyle=line_style,
            label=scheme,
        )
        last_date = pd.Timestamp(group["日期"].iloc[-1])
        last_value = float(cumulative.iloc[-1])
        axis.scatter(
            [last_date], [last_value],
            s=45, color=color, edgecolor="white", linewidth=1.0, zorder=6,
        )
        axis.annotate(
            f"{last_value:,.0f} 元",
            xy=(last_date, last_value),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=10,
            color=color,
            fontweight="bold",
            va="center",
            ha="left",
            zorder=7,
        )
    axis.set_xlabel("日期")
    axis.set_ylabel("累计总购电成本（元）")
    axis.set_ylim(bottom=0)
    _style_axes(axis)
    _format_year_axis(axis)
    axis.yaxis.set_major_formatter(mticker.FuncFormatter(_format_scientific_y))
    axis.set_xlim(right=end_date + pd.Timedelta(days=25))
    axis.legend(loc="upper left", ncol=3)
    _save_figure(figure, image_dir / versioned_name("图6_三种方案累计总购电成本对比", ".png"), dpi)

def plot_daily_emergency(daily: pd.DataFrame, image_dir: Path, dpi: int, formal_start: str) -> None:
    selected = daily.loc[pd.to_datetime(daily["日期"]) >= pd.Timestamp(formal_start)].copy()
    figure, axis = plt.subplots(figsize=(14, 6.5))
    styles = {
        SCHEME_RULE: (COLOR_GRAY, "-"),
        SCHEME_DETERMINISTIC: (COLOR_ORANGE, "-"),
        SCHEME_STOCHASTIC: (COLOR_BLUE, "-"),
    }
    for scheme in [SCHEME_RULE, SCHEME_DETERMINISTIC, SCHEME_STOCHASTIC]:
        group = selected.loc[selected["方案"].eq(scheme)].sort_values("日期")
        color, line_style = styles[scheme]
        axis.plot(group["日期"], group["紧急购电总量（kWh）"], color=color, linewidth=1.6, linestyle=line_style, label=scheme)
    axis.set_xlabel("日期")
    axis.set_ylabel("每日紧急购电量（kWh）")
    axis.set_ylim(bottom=0)
    _style_axes(axis)
    _format_year_axis(axis)
    axis.legend(loc="upper right", ncol=3)
    _save_figure(figure, image_dir / versioned_name("图7_三种方案每日紧急购电量对比", ".png"), dpi)


def build_monthly_mean_storage_series(detail: pd.DataFrame) -> pd.DataFrame:
    """计算主模型各月全部10分钟时段的平均期末储电量。"""
    main = detail.loc[detail["方案"].eq(SCHEME_STOCHASTIC)].copy()
    main["月份"] = pd.to_datetime(main["日期"]).dt.month
    return main.groupby("月份", as_index=False).agg(
        **{"月均储电量（kWh）": ("时段末储电量（kWh）", "mean")}
    )


def plot_monthly_mean_storage_energy(
    detail: pd.DataFrame,
    image_dir: Path,
    dpi: int,
    config: ModelConfig,
) -> None:
    monthly = build_monthly_mean_storage_series(detail)
    x = np.arange(len(monthly), dtype=float)
    figure, axis = plt.subplots(figsize=(14, 6.5))
    bars = axis.bar(
        x,
        monthly["月均储电量（kWh）"],
        width=0.62,
        color=COLOR_BLUE,
        alpha=0.82,
        edgecolor="#333333",
        linewidth=0.7,
        label="月均储电量",
    )
    upper_line = axis.axhline(config.maximum_storage_kwh, color="#E9CF72", linewidth=1.1, linestyle="--", label="储电量上限")
    lower_line = axis.axhline(config.minimum_storage_kwh, color="#A8D5A2", linewidth=1.1, linestyle="--", label="储电量下限")
    axis.set_xticks(x, [f"{month}月" for month in monthly["月份"]])
    axis.set_xlabel("月份")
    axis.set_ylabel("储电量（kWh）")
    axis.set_ylim(0, config.maximum_storage_kwh * 1.08)
    _style_axes(axis)
    axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=8)
    axis.legend([bars, upper_line, lower_line], ["月均储电量", "储电量上限", "储电量下限"], loc="upper right", ncol=3)
    _save_figure(figure, image_dir / versioned_name("图8_月均储能电量", ".png"), dpi)


def run_plot_stage(
    clean_long: pd.DataFrame,
    prediction: pd.DataFrame,
    detail: pd.DataFrame,
    daily: pd.DataFrame,
    image_dir: Path,
    config: ModelConfig,
    plot_date: str,
    dpi: int,
) -> None:
    configure_chinese_plot_style()
    _ = plot_date
    for data in [clean_long, prediction, detail, daily]:
        if "日期" in data.columns:
            data["日期"] = pd.to_datetime(data["日期"])
    obsolete_names = [
        versioned_name("图2_平均日内负载与光伏功率", ".png"),
        versioned_name("图3_负载与光伏预测效果", ".png"),
        versioned_name("图4_计划购电与实际净需求", ".png"),
        versioned_name("图5_储电量与实际充放电量", ".png"),
        versioned_name("图5_全年计划购电与实际净需求", ".png"),
        versioned_name("图8_全年储能电量", ".png"),
    ]
    for name in obsolete_names:
        path = image_dir / name
        if path.exists():
            path.unlink()
    plot_annual_energy(clean_long, image_dir, dpi)
    plot_daily_mean_boxplot(clean_long, image_dir, dpi)
    plot_annual_prediction_metrics(prediction, image_dir, dpi, config)
    plot_representative_prediction_days(prediction, image_dir, dpi)
    plot_monthly_mean_purchase_and_net_demand(detail, image_dir, dpi)
    plot_cumulative_cost(daily, image_dir, dpi, config.formal_start_date)
    plot_daily_emergency(daily, image_dir, dpi, config.formal_start_date)
    plot_monthly_mean_storage_energy(detail, image_dir, dpi, config)


def save_preprocessing_outputs(
    table_dir: Path,
    cache_dir: Path,
    quality: pd.DataFrame,
    anomalies: pd.DataFrame,
    before_after: pd.DataFrame,
    clean_long: pd.DataFrame,
) -> None:
    write_csv(quality, table_dir / versioned_name("数据质量报告", ".csv"))
    write_csv(anomalies, table_dir / versioned_name("异常记录清单", ".csv"))
    write_csv(before_after, table_dir / versioned_name("预处理前后对比", ".csv"))
    write_csv(clean_long, table_dir / versioned_name("附件2清洗长表", ".csv"))
    write_csv(clean_long, cache_dir / versioned_name("问题二年度实际数据绘图缓存", ".csv"))


def save_model_outputs(
    results: dict[str, pd.DataFrame],
    table_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    config: ModelConfig,
) -> None:
    detail = results["逐时段结果"]
    formal_start = pd.Timestamp(config.formal_start_date)
    formal_detail = detail.loc[pd.to_datetime(detail["日期"]) >= formal_start].copy()
    main_formal = formal_detail.loc[formal_detail["方案"].eq(SCHEME_STOCHASTIC)].copy()
    prediction_formal = results["预测结果"].loc[
        pd.to_datetime(results["预测结果"]["日期"]) >= formal_start
    ].copy()

    write_csv(main_formal, table_dir / versioned_name("问题二主模型逐时段结果", ".csv"))
    write_csv(formal_detail, table_dir / versioned_name("问题二三方案逐时段结果", ".csv"))
    write_csv(prediction_formal, table_dir / versioned_name("问题二预测结果", ".csv"))
    write_csv(results["预测指标"], table_dir / versioned_name("问题二预测指标", ".csv"))
    write_csv(results["情景抽样记录"], table_dir / versioned_name("问题二情景抽样记录", ".csv"))
    write_csv(results["求解日志"], table_dir / versioned_name("问题二求解日志", ".csv"))
    write_csv(results["每日汇总"], table_dir / versioned_name("问题二每日汇总", ".csv"))
    write_csv(results["年度对比"], table_dir / versioned_name("问题二全年汇总与基线对比", ".csv"))
    write_csv(results["约束校验"], table_dir / versioned_name("问题二约束校验", ".csv"))
    write_csv(results["模型参数"], table_dir / versioned_name("问题二模型参数", ".csv"))

    write_csv(results["预测结果"], cache_dir / versioned_name("问题二预测绘图缓存", ".csv"))
    write_csv(detail, cache_dir / versioned_name("问题二逐时段绘图缓存", ".csv"))
    write_csv(results["每日汇总"], cache_dir / versioned_name("问题二每日汇总绘图缓存", ".csv"))

    workbook_sheets = {
        "主模型逐时段结果": main_formal,
        "每日汇总": results["每日汇总"].loc[pd.to_datetime(results["每日汇总"]["日期"]) >= formal_start],
        "全年汇总与基线对比": results["年度对比"],
        "预测指标": results["预测指标"],
        "情景抽样记录": results["情景抽样记录"],
        "约束校验": results["约束校验"],
        "模型参数": results["模型参数"],
    }
    write_excel(workbook_sheets, output_dir / versioned_name("result2", ".xlsx"))


def load_plot_cache(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dir = output_dir / f"绘图缓存_{VERSION}"
    paths = {
        "clean": cache_dir / versioned_name("问题二年度实际数据绘图缓存", ".csv"),
        "prediction": cache_dir / versioned_name("问题二预测绘图缓存", ".csv"),
        "detail": cache_dir / versioned_name("问题二逐时段绘图缓存", ".csv"),
        "daily": cache_dir / versioned_name("问题二每日汇总绘图缓存", ".csv"),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("绘图模式缺少缓存文件：\n" + "\n".join(missing))
    return (
        pd.read_csv(paths["clean"], encoding="utf-8-sig", parse_dates=["日期"]),
        pd.read_csv(paths["prediction"], encoding="utf-8-sig", parse_dates=["日期"]),
        pd.read_csv(paths["detail"], encoding="utf-8-sig", parse_dates=["日期"]),
        pd.read_csv(paths["daily"], encoding="utf-8-sig", parse_dates=["日期"]),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="问题二滚动预测与两阶段随机线性规划（第一版）")
    parser.add_argument("--attachment1", default="附件1.xlsx", help="附件1路径")
    parser.add_argument("--attachment2", default="附件2.xlsx", help="附件2路径")
    parser.add_argument("--output-dir", default=f"问题二随机规划输出_{VERSION}", help="输出目录")
    parser.add_argument("--mode", choices=["all", "preprocess", "solve", "plot"], default="all", help="运行模式")
    parser.add_argument("--plot-date", default="2025-02-01", help="兼容旧命令保留，当前八张图不使用该参数")
    parser.add_argument("--scenario-count", type=int, default=50, help="联合残差情景数")
    parser.add_argument("--residual-window", type=int, default=60, help="有效联合残差滚动窗口天数")
    parser.add_argument("--seed", type=int, default=20250911, help="固定随机种子")
    parser.add_argument("--dpi", type=int, default=300, help="PNG图片分辨率")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = ModelConfig(
        scenario_count=args.scenario_count,
        residual_window_days=args.residual_window,
        random_seed=args.seed,
    )
    config.validate()
    output_dir = Path(args.output_dir)
    table_dir, cache_dir, image_dir = prepare_output_directories(output_dir)

    if args.mode == "plot":
        clean_long, prediction, detail, daily = load_plot_cache(output_dir)
        run_plot_stage(clean_long, prediction, detail, daily, image_dir, config, args.plot_date, args.dpi)
        return

    attachment1 = read_attachment1(Path(args.attachment1))
    dates, load_kw, pv_kw, endpoint = read_attachment2(Path(args.attachment2))
    quality, anomalies, before_after = build_quality_tables(dates, load_kw, pv_kw, endpoint)
    clean_long = build_clean_long_table(dates, load_kw, pv_kw, endpoint)
    save_preprocessing_outputs(table_dir, cache_dir, quality, anomalies, before_after, clean_long)

    if args.mode == "preprocess":
        configure_chinese_plot_style()
        plot_annual_energy(clean_long, image_dir, args.dpi)
        plot_daily_mean_boxplot(clean_long, image_dir, args.dpi)
        return

    results = run_annual_rolling_model(attachment1, dates, load_kw, pv_kw, endpoint, config)
    save_model_outputs(results, table_dir, cache_dir, output_dir, config)

    if args.mode == "all":
        run_plot_stage(
            clean_long,
            results["预测结果"],
            results["逐时段结果"],
            results["每日汇总"],
            image_dir,
            config,
            args.plot_date,
            args.dpi,
        )


if __name__ == "__main__":
    main()
