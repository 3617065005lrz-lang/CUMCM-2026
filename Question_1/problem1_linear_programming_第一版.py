"""
2026 CUMCM C题问题一：确定性线性规划求解代码（第一版）。

功能：
1. 读取附件1或上一阶段生成的清洗后CSV；
2. 建立144个10分钟时段的确定性线性规划；
3. 输出逐时段结果、论文表1、论文表2、Baseline对比和约束校验；
4. 输出result1_第一版.xlsx；
5. 保存绘图缓存，并生成四张中文结果图片。

环境：Python 3.11+
依赖：pandas、numpy、scipy、matplotlib、openpyxl

运行示例：
    py -3.11 problem1_linear_programming_第一版.py --input "附件1.xlsx" --mode all
    py -3.11 problem1_linear_programming_第一版.py --input "附件1.xlsx" --mode solve
    py -3.11 problem1_linear_programming_第一版.py --output-dir "问题一线性规划输出_第一版" --mode plot

模式：
    all    求解、输出表格并绘图
    solve  只求解并输出表格和绘图缓存
    plot   只读取绘图缓存重新生成四张图片
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import lil_matrix


VERSION = "第一版"
TIME_STEP_MINUTES = 10
EXPECTED_INTERVALS = 144

RAW_REQUIRED_COLUMNS = ["时间", "电价", "小区负载", "光伏发电预测功率"]
MODEL_REQUIRED_COLUMNS = [
    "时间段编号",
    "区间终点分钟",
    "时间段",
    "电价（元/kWh）",
    "负载电量（kWh）",
    "光伏电量（kWh）",
]


@dataclass(frozen=True)
class ModelConfig:
    """问题一线性规划参数。"""

    interval_hours: float = 1.0 / 6.0
    efficiency: float = 0.90
    initial_storage_kwh: float = 6000.0
    minimum_storage_kwh: float = 1200.0
    maximum_storage_kwh: float = 10800.0
    maximum_power_kw: float = 5000.0
    numerical_tolerance: float = 1e-7
    simultaneous_tolerance: float = 1e-6

    @property
    def maximum_interval_energy_kwh(self) -> float:
        return self.maximum_power_kw * self.interval_hours

    def validate(self) -> None:
        if not 0 < self.interval_hours <= 24:
            raise ValueError("时间步长必须大于0且不超过24小时。")
        if not 0 < self.efficiency <= 1:
            raise ValueError("充放电效率必须位于(0, 1]。")
        if self.minimum_storage_kwh > self.maximum_storage_kwh:
            raise ValueError("最小储电量不能大于最大储电量。")
        if not self.minimum_storage_kwh <= self.initial_storage_kwh <= self.maximum_storage_kwh:
            raise ValueError("初始储电量必须位于储电量上下限之间。")
        if self.maximum_power_kw <= 0:
            raise ValueError("最大充放电功率必须大于0。")


def versioned_name(stem: str, extension: str) -> str:
    extension = extension if extension.startswith(".") else f".{extension}"
    return f"{stem}_{VERSION}{extension}"


def format_clock(total_minutes: int) -> str:
    if total_minutes == 1440:
        return "24:00"
    hour, minute = divmod(total_minutes, 60)
    return f"{hour}:{minute:02d}"


def interval_label_from_endpoint(endpoint_minutes: int | float) -> str:
    if pd.isna(endpoint_minutes):
        return "时间无法解析"
    endpoint = int(endpoint_minutes)
    start = endpoint - TIME_STEP_MINUTES
    if start < 0 or endpoint > 1440:
        return "时间超出当天范围"
    return f"{format_clock(start)}-{format_clock(endpoint)}"


def parse_time_endpoint(value: Any) -> int | float:
    """把Excel时间对象或文本时间转换为区间右端点分钟数。"""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan

    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, time):
        return value.hour * 60 + value.minute

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return np.nan
        if 0 <= number <= 1:
            return int(round(number * 1440))
        if number.is_integer() and 0 <= number <= 1440:
            return int(number)
        raise ValueError(f"无法识别的数值时间：{value}")

    text = str(value).strip().replace("：", ":").replace(" ", "")
    if not text:
        return np.nan
    next_day = text.endswith("+1")
    if next_day:
        text = text[:-2]
    match = re.fullmatch(r"(\d{1,2}):(\d{1,2})(?::\d{1,2})?", text)
    if not match:
        raise ValueError(f"无法识别的文本时间：{value}")
    hour, minute = map(int, match.groups())
    if hour == 24 and minute == 0:
        return 1440
    if hour > 23 or minute > 59:
        raise ValueError(f"时间超出合理范围：{value}")
    endpoint = hour * 60 + minute
    if next_day:
        endpoint += 1440
    return endpoint


def _safe_parse_time(value: Any) -> int | float:
    try:
        return parse_time_endpoint(value)
    except (TypeError, ValueError):
        return np.nan


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result.columns = [str(column).strip() for column in result.columns]
    return result


def prepare_model_input(data: pd.DataFrame) -> pd.DataFrame:
    """把附件1原始字段或上一阶段清洗字段转换为统一模型输入。"""
    source = _normalize_columns(data)

    if all(column in source.columns for column in MODEL_REQUIRED_COLUMNS):
        model_data = source.loc[:, MODEL_REQUIRED_COLUMNS].copy()
        for column in ["区间终点分钟", "电价（元/kWh）", "负载电量（kWh）", "光伏电量（kWh）"]:
            model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
    elif all(column in source.columns for column in RAW_REQUIRED_COLUMNS):
        endpoint = source["时间"].map(_safe_parse_time)
        if len(endpoint) and endpoint.iloc[-1] == 0:
            endpoint.iloc[-1] = 1440
        price = pd.to_numeric(source["电价"], errors="coerce")
        load_power = pd.to_numeric(source["小区负载"], errors="coerce")
        pv_power = pd.to_numeric(source["光伏发电预测功率"], errors="coerce")
        model_data = pd.DataFrame(
            {
                "时间段编号": np.arange(1, len(source) + 1),
                "区间终点分钟": endpoint,
                "时间段": endpoint.map(interval_label_from_endpoint),
                "电价（元/kWh）": price,
                "负载电量（kWh）": load_power * (TIME_STEP_MINUTES / 60.0),
                "光伏电量（kWh）": pv_power * (TIME_STEP_MINUTES / 60.0),
            }
        )
    else:
        missing_raw = [column for column in RAW_REQUIRED_COLUMNS if column not in source.columns]
        missing_clean = [column for column in MODEL_REQUIRED_COLUMNS if column not in source.columns]
        raise ValueError(
            "输入文件既不符合附件1原始字段，也不符合清洗后字段。"
            f"\n原始字段缺少：{missing_raw}"
            f"\n清洗后字段缺少：{missing_clean}"
        )

    model_data = model_data.sort_values("区间终点分钟", kind="stable").reset_index(drop=True)
    model_data["时间段编号"] = np.arange(1, len(model_data) + 1)
    model_data["时间段"] = model_data["区间终点分钟"].map(interval_label_from_endpoint)
    validate_model_input(model_data)
    return model_data


def validate_model_input(data: pd.DataFrame, require_full_day: bool = True) -> None:
    missing_columns = [column for column in MODEL_REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"模型输入缺少字段：{missing_columns}")

    if require_full_day and len(data) != EXPECTED_INTERVALS:
        raise ValueError(f"附件1应包含{EXPECTED_INTERVALS}个时间段，实际为{len(data)}个。")

    numeric_columns = ["区间终点分钟", "电价（元/kWh）", "负载电量（kWh）", "光伏电量（kWh）"]
    missing_counts = data[numeric_columns].isna().sum()
    if int(missing_counts.sum()) > 0:
        detail = "、".join(f"{column}:{count}" for column, count in missing_counts.items() if count)
        raise ValueError(f"模型输入存在缺失或无法解析的数值：{detail}")

    if data["区间终点分钟"].duplicated().any():
        raise ValueError("模型输入存在重复时间段。")
    if require_full_day:
        expected = np.arange(TIME_STEP_MINUTES, 1441, TIME_STEP_MINUTES)
        actual = data["区间终点分钟"].to_numpy(dtype=int)
        if not np.array_equal(actual, expected):
            raise ValueError("时间段未完整覆盖0:00至24:00的144个连续10分钟区间。")

    if data["负载电量（kWh）"].lt(0).any():
        raise ValueError("小区耗电量不能为负。")
    if data["光伏电量（kWh）"].lt(0).any():
        raise ValueError("光伏发电量不能为负。")


def load_model_input(input_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(input_path, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(input_path, sheet_name=sheet_name or 0, engine="openpyxl")
    else:
        raise ValueError("输入文件只支持CSV、XLSX或XLS格式。")
    return prepare_model_input(raw)


def _clean_small_values(values: np.ndarray, tolerance: float) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result[np.abs(result) < tolerance] = 0.0
    return result


def solve_linear_program(
    model_input: pd.DataFrame,
    config: ModelConfig,
) -> tuple[pd.DataFrame, OptimizeResult]:
    """求解问题一确定性线性规划。"""
    config.validate()
    validate_model_input(model_input, require_full_day=False)
    data = model_input.reset_index(drop=True).copy()
    n = len(data)
    if n == 0:
        raise ValueError("模型输入不能为空。")

    price = data["电价（元/kWh）"].to_numpy(dtype=float)
    load = data["负载电量（kWh）"].to_numpy(dtype=float)
    photovoltaic = data["光伏电量（kWh）"].to_numpy(dtype=float)

    purchase_slice = slice(0, n)
    charge_slice = slice(n, 2 * n)
    discharge_slice = slice(2 * n, 3 * n)
    curtailment_slice = slice(3 * n, 4 * n)
    storage_slice = slice(4 * n, 5 * n + 1)
    variable_count = 5 * n + 1

    objective = np.zeros(variable_count)
    objective[purchase_slice] = price

    equality_matrix = lil_matrix((2 * n, variable_count), dtype=float)
    equality_rhs = np.zeros(2 * n, dtype=float)

    for period in range(n):
        energy_row = period
        equality_matrix[energy_row, purchase_slice.start + period] = 1.0
        equality_matrix[energy_row, charge_slice.start + period] = -1.0
        equality_matrix[energy_row, discharge_slice.start + period] = 1.0
        equality_matrix[energy_row, curtailment_slice.start + period] = -1.0
        equality_rhs[energy_row] = load[period] - photovoltaic[period]

        storage_row = n + period
        equality_matrix[storage_row, charge_slice.start + period] = -config.efficiency
        equality_matrix[storage_row, discharge_slice.start + period] = 1.0 / config.efficiency
        equality_matrix[storage_row, storage_slice.start + period] = -1.0
        equality_matrix[storage_row, storage_slice.start + period + 1] = 1.0

    maximum_interval_energy = config.maximum_interval_energy_kwh
    bounds: list[tuple[float | None, float | None]] = []
    bounds.extend([(0.0, None)] * n)
    bounds.extend([(0.0, maximum_interval_energy)] * n)
    bounds.extend([(0.0, maximum_interval_energy)] * n)
    bounds.extend([(0.0, float(photovoltaic[period])) for period in range(n)])
    for storage_index in range(n + 1):
        if storage_index in {0, n}:
            bounds.append((config.initial_storage_kwh, config.initial_storage_kwh))
        else:
            bounds.append((config.minimum_storage_kwh, config.maximum_storage_kwh))

    solver_result = linprog(
        c=objective,
        A_eq=equality_matrix.tocsr(),
        b_eq=equality_rhs,
        bounds=bounds,
        method="highs",
    )
    if not solver_result.success:
        raise RuntimeError(
            "线性规划未找到最优解。"
            f"求解状态：{solver_result.status}；求解信息：{solver_result.message}"
        )

    solution = _clean_small_values(solver_result.x, config.numerical_tolerance)
    purchase = solution[purchase_slice]
    charge = solution[charge_slice]
    discharge = solution[discharge_slice]
    curtailment = solution[curtailment_slice]
    storage = solution[storage_slice]

    result = data.copy()
    result["区间开始分钟"] = result["区间终点分钟"] - TIME_STEP_MINUTES
    result["小区耗电量（kWh）"] = load
    result["光伏发电量（kWh）"] = photovoltaic
    result["计划购电量（kWh）"] = purchase
    result["充电量（kWh）"] = charge
    result["放电量（kWh）"] = discharge
    result["弃光量（kWh）"] = curtailment
    result["时段开始储电量（kWh）"] = storage[:-1]
    result["时段结束储电量（kWh）"] = storage[1:]
    result["本时段购电费（元）"] = price * purchase
    result["能量平衡残差（kWh）"] = purchase + photovoltaic + discharge - load - charge - curtailment
    result["储能递推残差（kWh）"] = (
        storage[1:] - storage[:-1] - config.efficiency * charge + discharge / config.efficiency
    )
    result["是否同时充放电"] = np.where(
        (charge > config.simultaneous_tolerance) & (discharge > config.simultaneous_tolerance),
        "是",
        "否",
    )
    return result, solver_result


def build_paper_table_1(result: pd.DataFrame) -> pd.DataFrame:
    specified_starts = [600, 720, 840, 960, 1080, 1200]
    rows: list[list[Any]] = []
    for start in specified_starts:
        selected = result.loc[result["区间开始分钟"].eq(start)]
        if len(selected) != 1:
            raise ValueError(f"无法唯一定位指定时间段：{format_clock(start)}-{format_clock(start + 10)}")
        rows.append(
            [
                f"{format_clock(start)}-{format_clock(start + 10)}购电量",
                float(selected.iloc[0]["计划购电量（kWh）"]),
                "kWh",
            ]
        )
    rows.extend(
        [
            ["全天购电量", float(result["计划购电量（kWh）"].sum()), "kWh"],
            ["全天购电费", float(result["本时段购电费（元）"].sum()), "元"],
        ]
    )
    return pd.DataFrame(rows, columns=["指标", "数值", "单位"])


def build_paper_table_2(result: pd.DataFrame) -> pd.DataFrame:
    rows: list[list[Any]] = []
    for start in range(0, 24 * 60, 4 * 60):
        end = start + 4 * 60
        mask = result["区间开始分钟"].ge(start) & result["区间终点分钟"].le(end)
        rows.append(
            [
                f"{format_clock(start)}-{format_clock(end)}",
                float(result.loc[mask, "充电量（kWh）"].sum()),
                float(result.loc[mask, "放电量（kWh）"].sum()),
                np.nan,
            ]
        )
    rows.append(["0:00储电量", np.nan, np.nan, float(result.iloc[0]["时段开始储电量（kWh）"])])
    rows.append(["24:00储电量", np.nan, np.nan, float(result.iloc[-1]["时段结束储电量（kWh）"])])
    return pd.DataFrame(rows, columns=["时间段或指标", "充电量（kWh）", "放电量（kWh）", "储电量（kWh）"])


def build_baseline_comparison(result: pd.DataFrame) -> pd.DataFrame:
    load = result["小区耗电量（kWh）"]
    photovoltaic = result["光伏发电量（kWh）"]
    price = result["电价（元/kWh）"]
    baseline_purchase = (load - photovoltaic).clip(lower=0)
    baseline_curtailment = (photovoltaic - load).clip(lower=0)

    baseline_cost = float((price * baseline_purchase).sum())
    optimized_cost = float(result["本时段购电费（元）"].sum())
    total_photovoltaic = float(photovoltaic.sum())
    baseline_used_pv = total_photovoltaic - float(baseline_curtailment.sum())
    optimized_used_pv = total_photovoltaic - float(result["弃光量（kWh）"].sum())

    def utilization(used: float) -> float:
        return used / total_photovoltaic if total_photovoltaic > 0 else np.nan

    rows = [
        ["全天购电量", float(baseline_purchase.sum()), float(result["计划购电量（kWh）"].sum()), "kWh"],
        ["全天购电费", baseline_cost, optimized_cost, "元"],
        ["弃光量", float(baseline_curtailment.sum()), float(result["弃光量（kWh）"].sum()), "kWh"],
        ["光伏利用量", baseline_used_pv, optimized_used_pv, "kWh"],
        ["光伏利用率", utilization(baseline_used_pv), utilization(optimized_used_pv), "比例"],
        ["最大单时段购电量", float(baseline_purchase.max()), float(result["计划购电量（kWh）"].max()), "kWh"],
        ["平均单时段购电量", float(baseline_purchase.mean()), float(result["计划购电量（kWh）"].mean()), "kWh"],
        ["购电费节省额", 0.0, baseline_cost - optimized_cost, "元"],
        ["购电费下降比例", 0.0, (baseline_cost - optimized_cost) / baseline_cost if baseline_cost else np.nan, "比例"],
    ]
    comparison = pd.DataFrame(rows, columns=["指标", "无储能调度", "线性规划优化后", "单位"])
    comparison["变化值"] = comparison["线性规划优化后"] - comparison["无储能调度"]
    return comparison[["指标", "无储能调度", "线性规划优化后", "变化值", "单位"]]


def build_validation_table(
    result: pd.DataFrame,
    solver_result: OptimizeResult,
    config: ModelConfig,
) -> pd.DataFrame:
    tolerance = max(config.numerical_tolerance * 10, 1e-6)
    maximum_interval_energy = config.maximum_interval_energy_kwh
    purchase = result["计划购电量（kWh）"]
    charge = result["充电量（kWh）"]
    discharge = result["放电量（kWh）"]
    curtailment = result["弃光量（kWh）"]
    all_storage = np.r_[
        result.iloc[0]["时段开始储电量（kWh）"],
        result["时段结束储电量（kWh）"].to_numpy(),
    ]
    recalculated_cost = float((result["电价（元/kWh）"] * purchase).sum())
    objective_difference = abs(recalculated_cost - float(solver_result.fun))
    simultaneous_count = int(result["是否同时充放电"].eq("是").sum())

    checks: list[list[Any]] = []

    def add(item: str, actual: Any, requirement: str, passed: bool) -> None:
        checks.append([item, actual, requirement, "通过" if passed else "需复核"])

    add("求解状态", "最优解" if solver_result.success else "求解失败", "求解器成功返回最优解", bool(solver_result.success))
    add("最大能量平衡残差", float(result["能量平衡残差（kWh）"].abs().max()), f"不超过{tolerance}", result["能量平衡残差（kWh）"].abs().max() <= tolerance)
    add("最大储能递推残差", float(result["储能递推残差（kWh）"].abs().max()), f"不超过{tolerance}", result["储能递推残差（kWh）"].abs().max() <= tolerance)
    add("最小储电量", float(all_storage.min()), f"不低于{config.minimum_storage_kwh} kWh", all_storage.min() >= config.minimum_storage_kwh - tolerance)
    add("最大储电量", float(all_storage.max()), f"不高于{config.maximum_storage_kwh} kWh", all_storage.max() <= config.maximum_storage_kwh + tolerance)
    add("0:00储电量", float(all_storage[0]), f"等于{config.initial_storage_kwh} kWh", abs(all_storage[0] - config.initial_storage_kwh) <= tolerance)
    add("24:00储电量", float(all_storage[-1]), f"等于{config.initial_storage_kwh} kWh", abs(all_storage[-1] - config.initial_storage_kwh) <= tolerance)
    add("最大单时段充电量", float(charge.max()), f"不超过{maximum_interval_energy:.6f} kWh", charge.max() <= maximum_interval_energy + tolerance)
    add("最大单时段放电量", float(discharge.max()), f"不超过{maximum_interval_energy:.6f} kWh", discharge.max() <= maximum_interval_energy + tolerance)
    add("计划购电负值数量", int((purchase < -tolerance).sum()), "等于0", int((purchase < -tolerance).sum()) == 0)
    add("弃光负值数量", int((curtailment < -tolerance).sum()), "等于0", int((curtailment < -tolerance).sum()) == 0)
    add("同时充放电时段数", simultaneous_count, "等于0；若不为0则考虑混合整数规划", simultaneous_count == 0)
    add("购电费独立复算差值", objective_difference, f"不超过{tolerance}", objective_difference <= tolerance)
    return pd.DataFrame(checks, columns=["校验项目", "实际结果", "要求", "判定"])


def build_model_summary(
    result: pd.DataFrame,
    solver_result: OptimizeResult,
    config: ModelConfig,
) -> pd.DataFrame:
    rows = [
        ["模型名称", "问题一确定性线性规划", ""],
        ["求解方法", "HiGHS线性规划", ""],
        ["求解状态", "最优解" if solver_result.success else "求解失败", ""],
        ["时间段数量", len(result), "个"],
        ["时间步长", config.interval_hours, "小时"],
        ["充电效率", config.efficiency, "比例"],
        ["放电效率", config.efficiency, "比例"],
        ["最大充放电功率", config.maximum_power_kw, "kW"],
        ["最大单时段充放电量", config.maximum_interval_energy_kwh, "kWh"],
        ["最小储电量", config.minimum_storage_kwh, "kWh"],
        ["最大储电量", config.maximum_storage_kwh, "kWh"],
        ["0:00储电量", config.initial_storage_kwh, "kWh"],
        ["24:00储电量", float(result.iloc[-1]["时段结束储电量（kWh）"]), "kWh"],
        ["全天计划购电量", float(result["计划购电量（kWh）"].sum()), "kWh"],
        ["全天计划购电费", float(result["本时段购电费（元）"].sum()), "元"],
        ["全天充电量", float(result["充电量（kWh）"].sum()), "kWh"],
        ["全天放电量", float(result["放电量（kWh）"].sum()), "kWh"],
        ["全天弃光量", float(result["弃光量（kWh）"].sum()), "kWh"],
    ]
    return pd.DataFrame(rows, columns=["项目", "数值或说明", "单位"])


def prepare_output_directories(output_dir: Path) -> tuple[Path, Path, Path]:
    table_dir = output_dir / f"数据表_{VERSION}"
    cache_dir = output_dir / f"绘图缓存_{VERSION}"
    image_dir = output_dir / f"图片_{VERSION}"
    for directory in (table_dir, cache_dir, image_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return table_dir, cache_dir, image_dir


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.6f")


def _style_excel_workbook(path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    workbook = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = thin_border
        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.border = thin_border
                cell.alignment = Alignment(vertical="center")
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000"
        for column_cells in worksheet.columns:
            maximum_length = max(
                len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells
            )
            worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                max(maximum_length + 2, 10), 28
            )
    workbook.save(path)


def write_excel_result(
    output_path: Path,
    result: pd.DataFrame,
    paper_table_1: pd.DataFrame,
    paper_table_2: pd.DataFrame,
    comparison: pd.DataFrame,
    validation: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    plan_purchase = result[["时间段编号", "时间段", "计划购电量（kWh）", "本时段购电费（元）"]]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        plan_purchase.to_excel(writer, sheet_name="计划购电量", index=False)
        paper_table_2.to_excel(writer, sheet_name="充放电量", index=False)
        result.to_excel(writer, sheet_name="逐时段结果", index=False)
        paper_table_1.to_excel(writer, sheet_name="论文表1", index=False)
        comparison.to_excel(writer, sheet_name="优化前后对比", index=False)
        validation.to_excel(writer, sheet_name="约束校验", index=False)
        summary.to_excel(writer, sheet_name="模型摘要", index=False)
    _style_excel_workbook(output_path)


def configure_chinese_plot_style() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.sans-serif": [selected],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#B8C2CC",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "grid.alpha": 0.25,
        }
    )


def _set_time_axis(axis: plt.Axes) -> None:
    major_hours = np.arange(0, 25, 2)
    axis.set_xlim(0, 24)
    axis.set_xticks(major_hours)
    axis.set_xticklabels([f"{hour}:00" for hour in major_hours])
    axis.set_xlabel("时刻")


def _set_interval_index_axis(axis: plt.Axes) -> None:
    """横坐标内部使用1至144连续编号，刻度文字仍显示对应时刻。"""
    tick_positions = np.arange(0.5, EXPECTED_INTERVALS + 1.0, 12.0)
    tick_labels = [f"{hour}:00" for hour in range(0, 25, 2)]
    axis.set_xlim(0.5, EXPECTED_INTERVALS + 0.5)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels)
    axis.set_xlabel("时刻")


def draw_storage_step(
    axis: plt.Axes,
    storage_x: np.ndarray,
    storage_y: np.ndarray,
) -> plt.Line2D:
    """按1至144连续时段编号绘制储电量折线。"""
    return axis.plot(
        storage_x,
        storage_y,
        color="#1F3A93",
        linewidth=2.0,
        label="储电量",
        zorder=4,
    )[0]


def _interval_step_data(result: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """按预处理后的顺序生成1至144连续时段横坐标。"""
    x = np.arange(1, len(result) + 1, dtype=float)
    values = result[column].to_numpy(dtype=float)
    return x, values


def _style_time_series_axis(axis: plt.Axes, use_interval_index: bool = False) -> None:
    """统一日内时序图的坐标轴与网格样式。"""
    if use_interval_index:
        _set_interval_index_axis(axis)
    else:
        _set_time_axis(axis)
    axis.set_ylim(bottom=0)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#A7ADB5", linestyle="--", linewidth=0.5, alpha=0.45)
    axis.spines["top"].set_visible(False)
    axis.margins(x=0)


def _annotate_maximum(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    unit: str,
) -> None:
    """仅标注长时间序列的最高点，避免给全部数据点加标记。"""
    source_x = np.asarray(x, dtype=float)
    source_y = np.asarray(y, dtype=float)
    if source_y.size == 0 or not np.isfinite(source_y).any():
        return
    index = int(np.nanargmax(source_y))
    maximum = float(source_y[index])
    if maximum <= 0:
        return
    point_x = float(source_x[index])
    horizontal_offset = -105 if point_x >= axis.get_xlim()[1] * 0.8 else 10
    axis.scatter([point_x], [maximum], s=28, color=color, edgecolor="white", linewidth=0.8, zorder=7)
    axis.annotate(
        f"最高值：{maximum:.2f} {unit}",
        xy=(point_x, maximum),
        xytext=(horizontal_offset, -30),
        textcoords="offset points",
        fontsize=9,
        color=color,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8},
        zorder=8,
    )


def _annotate_discharge_maximum(
    axis: plt.Axes,
    x: np.ndarray,
    signed_discharge: np.ndarray,
    color: str,
) -> None:
    """在负半轴标注绝对值最大的放电量。"""
    source_x = np.asarray(x, dtype=float)
    source_y = np.asarray(signed_discharge, dtype=float)
    if source_y.size == 0 or not np.isfinite(source_y).any():
        return
    index = int(np.nanargmin(source_y))
    minimum = float(source_y[index])
    if minimum >= 0:
        return
    point_x = float(source_x[index])
    horizontal_offset = -125 if point_x >= axis.get_xlim()[1] * 0.8 else 10
    axis.scatter([point_x], [minimum], s=28, color=color, edgecolor="white", linewidth=0.8, zorder=7)
    axis.annotate(
        f"最大放电量：{-minimum:.2f} kWh",
        xy=(point_x, minimum),
        xytext=(horizontal_offset, 24),
        textcoords="offset points",
        fontsize=9,
        color=color,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": color, "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.8},
        zorder=8,
    )


def _finish_figure(figure: plt.Figure, output_path: Path, dpi: int, bottom: float = 0.07) -> None:
    figure.tight_layout(rect=(0, bottom, 1, 1))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_price_purchase_figure(result: pd.DataFrame, image_dir: Path, dpi: int) -> None:
    time_x, price_y = _interval_step_data(result, "电价（元/kWh）")
    _, purchase_y = _interval_step_data(result, "计划购电量（kWh）")

    figure, price_axis = plt.subplots(figsize=(14, 6.5))
    purchase_axis = price_axis.twinx()

    price_line = price_axis.plot(
        time_x,
        price_y,
        color="#D95F02",
        linestyle="-",
        linewidth=2.0,
        label="电价",
        zorder=4,
    )[0]
    purchase_line = purchase_axis.plot(
        time_x,
        purchase_y,
        color="#1F3A93",
        linestyle="-",
        linewidth=2.0,
        label="计划购电量",
        zorder=3,
    )[0]

    price_axis.set_title("电价与计划购电量的日内变化")
    price_axis.set_ylabel("电价（元/kWh）", color="#D95F02")
    purchase_axis.set_ylabel("每10分钟购电量（kWh）", color="#1F3A93")
    price_axis.tick_params(axis="y", labelcolor="#D95F02")
    purchase_axis.tick_params(axis="y", labelcolor="#1F3A93")
    _style_time_series_axis(price_axis, use_interval_index=True)
    purchase_axis.set_ylim(bottom=0)
    purchase_axis.spines["top"].set_visible(False)

    _annotate_maximum(price_axis, time_x, price_y, "#D95F02", "元/kWh")
    _annotate_maximum(purchase_axis, time_x, purchase_y, "#1F3A93", "kWh")
    price_axis.legend(
        [price_line, purchase_line],
        ["电价", "计划购电量"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    _finish_figure(
        figure,
        image_dir / versioned_name("图1_电价与计划购电量", ".png"),
        dpi,
    )


def save_storage_discharge_figure(
    result: pd.DataFrame,
    image_dir: Path,
    config: ModelConfig,
    dpi: int,
) -> None:
    flow_x, charge_y = _interval_step_data(result, "充电量（kWh）")
    _, discharge_y = _interval_step_data(result, "放电量（kWh）")
    signed_discharge_y = -discharge_y
    storage_x = np.arange(1, len(result) + 1, dtype=float)
    storage_y = result["时段结束储电量（kWh）"].to_numpy(dtype=float)

    figure, storage_axis = plt.subplots(figsize=(14, 6.5))
    discharge_axis = storage_axis.twinx()

    storage_line = draw_storage_step(storage_axis, storage_x, storage_y)
    charge_line = discharge_axis.plot(
        flow_x,
        charge_y,
        color="#1B9E77",
        linestyle="-",
        linewidth=2.0,
        label="充电量",
        zorder=3,
    )[0]
    discharge_line = discharge_axis.plot(
        flow_x,
        signed_discharge_y,
        color="#D95F02",
        linestyle="-",
        linewidth=2.0,
        label="放电量",
        zorder=3,
    )[0]
    rated_lower_line = storage_axis.axhline(
        1200.0,
        color="#A8D5A2",
        linestyle="--",
        linewidth=1.0,
        label="额定下限",
    )
    rated_upper_line = storage_axis.axhline(
        10800.0,
        color="#E9CF72",
        linestyle="--",
        linewidth=1.0,
        label="额定上限",
    )
    initial_line = storage_axis.axhline(
        config.initial_storage_kwh,
        color="#F4A6A6",
        linestyle="--",
        linewidth=1.0,
        label="初始储电量",
    )
    nominal_capacity_line = storage_axis.axhline(
        12000.0,
        color="#A7ADB5",
        linestyle="-.",
        linewidth=1.0,
        label="额定容量",
    )

    storage_axis.set_title("储电量与充放电量的日内变化")
    storage_axis.set_ylabel("储电量（kWh）", color="#1F3A93")
    discharge_axis.set_ylabel("每10分钟充放电量（kWh）")
    storage_axis.tick_params(axis="y", labelcolor="#1F3A93")
    discharge_axis.tick_params(axis="y", labelcolor="#52657A")
    _style_time_series_axis(storage_axis, use_interval_index=True)
    storage_axis.set_ylim(0, 12000.0)
    storage_axis.set_yticks(np.arange(0, 12001, 2000))
    flow_limit = max(float(np.nanmax(charge_y)), float(np.nanmax(discharge_y)), 1.0) * 1.15
    discharge_axis.set_ylim(-flow_limit, flow_limit)
    discharge_axis.axhline(0, color="#A7ADB5", linewidth=0.5, zorder=1)
    discharge_axis.spines["top"].set_visible(False)

    _annotate_maximum(storage_axis, storage_x, storage_y, "#1F3A93", "kWh")
    _annotate_maximum(discharge_axis, flow_x, charge_y, "#1B9E77", "kWh")
    _annotate_discharge_maximum(discharge_axis, flow_x, signed_discharge_y, "#D95F02")
    handles = [
        storage_line,
        charge_line,
        discharge_line,
        rated_lower_line,
        rated_upper_line,
        initial_line,
        nominal_capacity_line,
    ]
    labels = [handle.get_label() for handle in handles]
    storage_axis.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    _finish_figure(
        figure,
        image_dir / versioned_name("图2_储电量与充放电量", ".png"),
        dpi,
    )


def save_photovoltaic_load_figure(result: pd.DataFrame, image_dir: Path, dpi: int) -> None:
    x, load = _interval_step_data(result, "小区耗电量（kWh）")
    _, photovoltaic = _interval_step_data(result, "光伏发电量（kWh）")

    figure, axis = plt.subplots(figsize=(14, 6.5))
    load_line = axis.plot(
        x,
        load,
        color="#1F3A93",
        linestyle="-",
        linewidth=2.0,
        label="小区耗电量",
    )[0]
    photovoltaic_line = axis.plot(
        x,
        photovoltaic,
        color="#D95F02",
        linestyle="-",
        linewidth=2.0,
        label="光伏发电量",
    )[0]
    axis.set_title("光伏发电量与小区耗电量的日内变化")
    axis.set_ylabel("每10分钟电量（kWh）")
    _style_time_series_axis(axis, use_interval_index=True)
    _annotate_maximum(axis, x, load, "#1F3A93", "kWh")
    _annotate_maximum(axis, x, photovoltaic, "#D95F02", "kWh")
    axis.legend(
        [load_line, photovoltaic_line],
        ["小区耗电量", "光伏发电量"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    _finish_figure(
        figure,
        image_dir / versioned_name("图3_光伏发电量与小区耗电量", ".png"),
        dpi,
    )


def build_cumulative_cost_series(result: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算无储能基准方案与线性规划方案的累计购电费用。"""
    x = np.arange(len(result) + 1, dtype=float) * TIME_STEP_MINUTES / 60.0
    price = result["电价（元/kWh）"].to_numpy(dtype=float)
    load = result["小区耗电量（kWh）"].to_numpy(dtype=float)
    photovoltaic = result["光伏发电量（kWh）"].to_numpy(dtype=float)
    purchase = result["计划购电量（kWh）"].to_numpy(dtype=float)
    baseline_interval_cost = price * np.maximum(load - photovoltaic, 0.0)
    optimized_interval_cost = price * purchase
    baseline_cumulative = np.r_[0.0, np.cumsum(baseline_interval_cost)]
    optimized_cumulative = np.r_[0.0, np.cumsum(optimized_interval_cost)]
    return x, baseline_cumulative, optimized_cumulative


def save_cumulative_cost_figure(result: pd.DataFrame, image_dir: Path, dpi: int) -> None:
    x, baseline_cost, optimized_cost = build_cumulative_cost_series(result)
    figure, axis = plt.subplots(figsize=(14, 6.5))

    baseline_line = axis.plot(
        x,
        baseline_cost,
        color="#D95F02",
        linestyle="-",
        linewidth=2.0,
        label="无储能调度",
    )[0]
    optimized_line = axis.plot(
        x,
        optimized_cost,
        color="#1F3A93",
        linestyle="-",
        linewidth=2.0,
        label="线性规划优化后",
    )[0]

    axis.set_title("优化前后累计购电费用对比")
    axis.set_ylabel("累计购电费用（元）")
    _style_time_series_axis(axis)
    axis.scatter([x[-1]], [baseline_cost[-1]], s=30, color="#D95F02", edgecolor="white", linewidth=0.8, zorder=5)
    axis.scatter([x[-1]], [optimized_cost[-1]], s=30, color="#1F3A93", edgecolor="white", linewidth=0.8, zorder=5)

    saving = float(baseline_cost[-1] - optimized_cost[-1])
    saving_rate = saving / float(baseline_cost[-1]) if baseline_cost[-1] else 0.0
    axis.annotate(
        f"全天节省：{saving:.2f} 元（{saving_rate:.2%}）",
        xy=(x[-1], optimized_cost[-1]),
        xytext=(-190, -42 if saving >= 0 else 28),
        textcoords="offset points",
        fontsize=9,
        color="#1F3A93",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#1F3A93", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "#1F3A93", "linewidth": 0.8},
    )
    axis.legend(
        [baseline_line, optimized_line],
        ["无储能调度", "线性规划优化后"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    _finish_figure(
        figure,
        image_dir / versioned_name("图4_优化前后累计购电费用", ".png"),
        dpi,
    )


def save_plot_cache(result: pd.DataFrame, cache_dir: Path) -> Path:
    columns = [
        "时间段编号",
        "区间开始分钟",
        "区间终点分钟",
        "时间段",
        "电价（元/kWh）",
        "时段开始储电量（kWh）",
        "时段结束储电量（kWh）",
        "计划购电量（kWh）",
        "充电量（kWh）",
        "放电量（kWh）",
        "光伏发电量（kWh）",
        "小区耗电量（kWh）",
    ]
    cache_path = cache_dir / versioned_name("问题一绘图缓存", ".csv")
    write_csv(result[columns], cache_path)
    return cache_path


def load_plot_cache(output_dir: Path) -> pd.DataFrame:
    _, cache_dir, _ = prepare_output_directories(output_dir)
    cache_path = cache_dir / versioned_name("问题一绘图缓存", ".csv")
    if not cache_path.exists():
        raise FileNotFoundError(f"缺少绘图缓存：{cache_path}。请先运行solve或all模式。")
    return pd.read_csv(cache_path, encoding="utf-8-sig")


def run_plot_stage(result: pd.DataFrame, output_dir: Path, config: ModelConfig, dpi: int) -> None:
    _, _, image_dir = prepare_output_directories(output_dir)
    configure_chinese_plot_style()
    obsolete_names = [
        versioned_name("图1_储电量计划购电量与放电量", ".png"),
        versioned_name("图2_光伏发电量与小区耗电量", ".png"),
        versioned_name("图2_储电量与放电量", ".png"),
    ]
    for obsolete_name in obsolete_names:
        obsolete_path = image_dir / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()

    save_price_purchase_figure(result, image_dir, dpi)
    save_storage_discharge_figure(result, image_dir, config, dpi)
    save_photovoltaic_load_figure(result, image_dir, dpi)
    save_cumulative_cost_figure(result, image_dir, dpi)


def run_solve_stage(
    input_path: Path,
    output_dir: Path,
    sheet_name: str | None,
    config: ModelConfig,
) -> pd.DataFrame:
    model_input = load_model_input(input_path, sheet_name)
    result, solver_result = solve_linear_program(model_input, config)

    paper_table_1 = build_paper_table_1(result)
    paper_table_2 = build_paper_table_2(result)
    comparison = build_baseline_comparison(result)
    validation = build_validation_table(result, solver_result, config)
    summary = build_model_summary(result, solver_result, config)

    table_dir, cache_dir, _ = prepare_output_directories(output_dir)
    outputs = [
        (result, "问题一逐时段优化结果"),
        (paper_table_1, "论文表1_购电量与购电费"),
        (paper_table_2, "论文表2_储能充放电量"),
        (comparison, "优化前后对比"),
        (validation, "问题一约束校验"),
        (summary, "问题一模型摘要"),
    ]
    for data, stem in outputs:
        write_csv(data, table_dir / versioned_name(stem, ".csv"))

    save_plot_cache(result, cache_dir)
    write_excel_result(
        output_dir / versioned_name("result1", ".xlsx"),
        result,
        paper_table_1,
        paper_table_2,
        comparison,
        validation,
        summary,
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C题问题一确定性线性规划求解")
    parser.add_argument("--input", default="附件1.xlsx", help="附件1或清洗后CSV路径")
    parser.add_argument("--output-dir", default=f"问题一线性规划输出_{VERSION}", help="输出文件夹")
    parser.add_argument("--sheet", default=None, help="Excel工作表；默认读取第一个工作表")
    parser.add_argument("--mode", choices=["all", "solve", "plot"], default="all", help="运行模式")
    parser.add_argument("--efficiency", type=float, default=0.90, help="充电效率和放电效率")
    parser.add_argument("--initial-storage", type=float, default=6000.0, help="0:00和24:00储电量，单位kWh")
    parser.add_argument("--minimum-storage", type=float, default=1200.0, help="最小储电量，单位kWh")
    parser.add_argument("--maximum-storage", type=float, default=10800.0, help="最大储电量，单位kWh")
    parser.add_argument("--maximum-power", type=float, default=5000.0, help="最大充放电功率，单位kW")
    parser.add_argument("--dpi", type=int, default=300, help="图片分辨率")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    config = ModelConfig(
        efficiency=args.efficiency,
        initial_storage_kwh=args.initial_storage,
        minimum_storage_kwh=args.minimum_storage,
        maximum_storage_kwh=args.maximum_storage,
        maximum_power_kw=args.maximum_power,
    )

    if args.mode in {"all", "solve"}:
        result = run_solve_stage(Path(args.input), output_dir, args.sheet, config)
    else:
        result = load_plot_cache(output_dir)

    if args.mode in {"all", "plot"}:
        run_plot_stage(result, output_dir, config, args.dpi)

    print(f"处理完成。输出位置：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
