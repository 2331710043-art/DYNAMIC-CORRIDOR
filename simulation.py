"""
Bo may mo phong Hanh lang dong (Dynamic Priority Corridor)
------------------------------------------------------------
Trien khai lai thuat toan mo ta trong de tai:
"Ung dung ly thuyet hang doi trong cau hinh duong cat ha canh dong
tai Cang HKQT Tan Son Nhat" (Chuong 3 & 4).

Mo hinh:
- Luong den (Arrival) va luong di (Departure) la 2 tien trinh Poisson
  doc lap, toc do lambda_arr, lambda_dep (chuyen/gio).
- Thoi gian phuc vu (ROT) cua tung luong duoc gia dinh la hang so trung
  binh (service_arr, service_dep - giay/chuyen), dong vai tro thoi gian
  chiem dung duong bang.
- Mo phong theo co che roi rac theo chu ky (time-step simulation): moi
  chu ky co do dai cycle_length (phut). O moi chu ky:
    1. Sinh may bay moi bang ky thuat "bo tich luy phan du" (residual
       accumulator) de tranh sai lech do lam tron.
    2. Xac dinh su kien khan cap (Emergency) neu co, chiem dung duong
       bang uu tien tuyet doi trong Emergency Service Time.
    3. Xac dinh huong uu tien phuc vu (chi danh cho Mixed mode) theo
       thuat toan Hanh lang dong: so sanh do dai hang doi 2 luong, chi
       chuyen doi uu tien khi da giu che do hien tai toi thieu Minimum
       Hold Time (tranh dao chieu uu tien lien tuc).
    4. Phuc vu may bay bang co che "dong ho may chu" (server clock):
       thoi diem duong bang ranh tro lai duoc cong don qua cac chu ky,
       nho do mo hinh xu ly dung ca truong hop ROT dai hon 1 chu ky ma
       khong lam hang doi tang ao do sai so lam tron nang luc.
- Ket qua tung chu ky duoc luu lai de tinh cac chi so hieu nang
  (Lq, Wq, rho, throughput...) va doi chieu voi cong thuc ly thuyet
  Pollaczek-Khinchine (M/G/1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Cau hinh dau vao
# --------------------------------------------------------------------------- #
@dataclass
class SimConfig:
    simulation_time: float = 60.0      # phut - tong thoi gian mo phong
    cycle_length: float = 1.0          # phut - do dai 1 chu ky
    arrival_rate: float = 21.0         # chuyen/gio (lambda_arr)
    departure_rate: float = 21.0       # chuyen/gio (lambda_dep)
    service_arr: float = 75.0          # giay/chuyen - ROT trung binh (den)
    service_dep: float = 85.0          # giay/chuyen - ROT trung binh (di)
    runway_mode: str = "mixed"         # "arr_only" | "dep_only" | "mixed"
    min_hold_time: float = 5.0         # phut - chi ap dung khi mixed
    emergency_prob: float = 0.0        # xac suat / chu ky, [0, 1]
    emergency_service_time: float = 180.0  # giay - thoi gian xu ly khan cap
    policy: str = "dynamic"            # "dynamic" | "fcfs" (dung de so sanh)
    seed: Optional[int] = None

    @property
    def n_cycles(self) -> int:
        return max(1, round(self.simulation_time / self.cycle_length))


# --------------------------------------------------------------------------- #
# Ket qua mo phong
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    cycles: list = field(default_factory=list)
    time_min: list = field(default_factory=list)
    lq_arr: list = field(default_factory=list)
    lq_dep: list = field(default_factory=list)
    served_arr: list = field(default_factory=list)
    served_dep: list = field(default_factory=list)
    mode_active: list = field(default_factory=list)
    utilization: list = field(default_factory=list)
    emergency: list = field(default_factory=list)
    cum_arrived: list = field(default_factory=list)
    cum_departed_demand: list = field(default_factory=list)
    cum_served: list = field(default_factory=list)

    summary: dict = field(default_factory=dict)

    def to_records(self) -> list:
        keys = [
            "cycles", "time_min", "lq_arr", "lq_dep", "served_arr",
            "served_dep", "mode_active", "utilization", "emergency",
        ]
        n = len(self.cycles)
        return [{k: getattr(self, k)[i] for k in keys} for i in range(n)]


# --------------------------------------------------------------------------- #
# Loi giai ly thuyet M/G/1 (Pollaczek-Khinchine)
# --------------------------------------------------------------------------- #
def pk_theoretical(lam_per_hour: float, mu_per_hour: float, sigma_hour: float) -> dict:
    """
    Cong thuc 2.3 - 2.6 trong de tai:
        Lq = (rho^2 + lambda^2 * sigma^2) / (2 * (1 - rho))
        L  = Lq + rho
        Wq = Lq / lambda   (gio)
        W  = Wq + 1/mu

    lambda, mu: chuyen/gio. Tra ve Wq, W theo don vi PHUT de de doi chieu
    voi ket qua mo phong. Neu he thong khong on dinh (rho >= 1) tra ve
    None cho cac gia tri phu thuoc (giong "diem gay" duoc mo ta trong
    bao cao khi lambda > mu).
    """
    if mu_per_hour <= 0:
        return {"rho": None, "Lq": None, "L": None, "Wq_min": None, "W_min": None}

    rho = lam_per_hour / mu_per_hour
    if rho >= 1:
        return {"rho": rho, "Lq": None, "L": None, "Wq_min": None, "W_min": None}

    Lq = (rho ** 2 + (lam_per_hour ** 2) * (sigma_hour ** 2)) / (2 * (1 - rho))
    L = Lq + rho
    Wq_hour = Lq / lam_per_hour if lam_per_hour > 0 else 0.0
    W_hour = Wq_hour + 1.0 / mu_per_hour

    return {
        "rho": rho,
        "Lq": Lq,
        "L": L,
        "Wq_min": Wq_hour * 60.0,
        "W_min": W_hour * 60.0,
    }


# --------------------------------------------------------------------------- #
# Dong co mo phong chinh
# --------------------------------------------------------------------------- #
def run_simulation(cfg: SimConfig) -> SimResult:
    rng = np.random.default_rng(cfg.seed)

    dt_min = cfg.cycle_length
    dt_sec = dt_min * 60.0
    n = cfg.n_cycles

    mean_arr_per_cycle = cfg.arrival_rate * dt_min / 60.0
    mean_dep_per_cycle = cfg.departure_rate * dt_min / 60.0

    residual_arr = 0.0
    residual_dep = 0.0

    queue_arr = 0.0
    queue_dep = 0.0

    # trang thai che do uu tien (chi dung khi Mixed + Dynamic)
    current_priority = "arr"
    cycles_in_mode = 0
    min_hold_cycles = max(1, round(cfg.min_hold_time / dt_min))

    # "dong ho may chu": thoi diem (giay, tinh tu t=0) duong bang se ranh
    # tro lai. Co che nay xu ly dung ca truong hop thoi gian phuc vu (ROT)
    # dai hon mot chu ky, tranh viec lam tron nang luc rieng le tung chu
    # ky lam hang doi phinh to gia tao.
    free_at_sec = 0.0

    result = SimResult()

    cum_arrived = 0
    cum_dep_demand = 0
    cum_served = 0

    for c in range(n):
        t_start = c * dt_sec
        t_end = (c + 1) * dt_sec

        # ------------------------------------------------------------- #
        # 1) Sinh may bay moi - ky thuat bo tich luy phan du
        # ------------------------------------------------------------- #
        arr_draw = int(rng.poisson(mean_arr_per_cycle)) if mean_arr_per_cycle > 0 else 0
        dep_draw = int(rng.poisson(mean_dep_per_cycle)) if mean_dep_per_cycle > 0 else 0

        residual_arr += mean_arr_per_cycle - arr_draw
        residual_dep += mean_dep_per_cycle - dep_draw
        extra_arr = 0
        extra_dep = 0
        if residual_arr >= 1.0:
            extra_arr = int(residual_arr)
            residual_arr -= extra_arr
        if residual_dep >= 1.0:
            extra_dep = int(residual_dep)
            residual_dep -= extra_dep

        new_arr = max(0, arr_draw + extra_arr)
        new_dep = max(0, dep_draw + extra_dep)

        queue_arr += new_arr
        queue_dep += new_dep
        cum_arrived += new_arr
        cum_dep_demand += new_dep

        # ------------------------------------------------------------- #
        # 2) Su kien khan cap - chiem dung duong bang uu tien tuyet doi
        # ------------------------------------------------------------- #
        is_emergency = bool(rng.random() < cfg.emergency_prob) if cfg.emergency_prob > 0 else False
        if is_emergency:
            free_at_sec = max(free_at_sec, t_start) + cfg.emergency_service_time

        # ------------------------------------------------------------- #
        # 3) Xac dinh huong uu tien phuc vu trong chu ky (thuat toan
        #    Hanh lang dong / hoac baseline FCFS de so sanh)
        # ------------------------------------------------------------- #
        mode = cfg.runway_mode
        if mode == "arr_only":
            priority_first, priority_second = "arr", None
        elif mode == "dep_only":
            priority_first, priority_second = "dep", None
        else:  # mixed
            if cfg.policy == "fcfs":
                # baseline doi chung: luan phien co dinh 1:1, khong xet
                # nguong hang doi (tuong tu triet ly MCBS/FCFS truyen thong)
                priority_first = "arr" if c % 2 == 0 else "dep"
                priority_second = "dep" if priority_first == "arr" else "arr"
            else:
                desired = "dep" if queue_dep > queue_arr else "arr"
                if desired != current_priority and cycles_in_mode >= min_hold_cycles:
                    current_priority = desired
                    cycles_in_mode = 0
                cycles_in_mode += 1
                priority_first = current_priority
                priority_second = "dep" if priority_first == "arr" else "arr"

        # ------------------------------------------------------------- #
        # 4) Phuc vu may bay bang co che dong ho may chu (server clock)
        # ------------------------------------------------------------- #
        served_arr = 0
        served_dep = 0

        def _queue_len(direction):
            return queue_arr if direction == "arr" else queue_dep

        def _svc_time(direction):
            return cfg.service_arr if direction == "arr" else cfg.service_dep

        guard = 0
        max_iter = 100000
        while free_at_sec < t_end and guard < max_iter:
            guard += 1
            direction = None
            if priority_first is not None and _queue_len(priority_first) > 0:
                direction = priority_first
            elif priority_second is not None and _queue_len(priority_second) > 0:
                direction = priority_second

            if direction is None:
                break

            svc = _svc_time(direction)
            if svc <= 0:
                break

            start = max(free_at_sec, t_start)
            if start >= t_end:
                break

            free_at_sec = start + svc
            if direction == "arr":
                queue_arr -= 1
                served_arr += 1
            else:
                queue_dep -= 1
                served_dep += 1

        cum_served += served_arr + served_dep

        # Muc su dung (utilization) cua chu ky: phan thoi gian duong bang
        # ban trong khoang [t_start, t_end). Dung dong ho may chu de tinh
        # chinh xac ca phan "no" tu chu ky truoc chuyen sang.
        busy_in_cycle = min(free_at_sec, t_end) - t_start
        utilization = max(0.0, min(1.0, busy_in_cycle / dt_sec)) if dt_sec > 0 else 0.0

        # ------------------------------------------------------------- #
        # 5) Ghi nhan ket qua chu ky
        # ------------------------------------------------------------- #
        result.cycles.append(c + 1)
        result.time_min.append(round((c + 1) * dt_min, 4))
        result.lq_arr.append(queue_arr)
        result.lq_dep.append(queue_dep)
        result.served_arr.append(served_arr)
        result.served_dep.append(served_dep)
        result.mode_active.append(priority_first if mode == "mixed" else mode)
        result.utilization.append(utilization)
        result.emergency.append(is_emergency)
        result.cum_arrived.append(cum_arrived)
        result.cum_departed_demand.append(cum_dep_demand)
        result.cum_served.append(cum_served)

    # ------------------------------------------------------------------- #
    # Tong hop chi so hieu nang (KPI) sau khi mo phong ket thuc
    # ------------------------------------------------------------------- #
    total_hours = cfg.simulation_time / 60.0
    lam_total = cfg.arrival_rate + cfg.departure_rate
    avg_lq_arr = float(np.mean(result.lq_arr)) if result.lq_arr else 0.0
    avg_lq_dep = float(np.mean(result.lq_dep)) if result.lq_dep else 0.0
    avg_lq_total = avg_lq_arr + avg_lq_dep
    avg_util = float(np.mean(result.utilization)) if result.utilization else 0.0

    lam_arr_h = cfg.arrival_rate
    lam_dep_h = cfg.departure_rate
    wq_arr_min = (avg_lq_arr / lam_arr_h) * 60.0 if lam_arr_h > 0 else 0.0
    wq_dep_min = (avg_lq_dep / lam_dep_h) * 60.0 if lam_dep_h > 0 else 0.0
    wq_total_min = (avg_lq_total / lam_total) * 60.0 if lam_total > 0 else 0.0

    throughput_per_h = (cum_served / total_hours) if total_hours > 0 else 0.0
    n_emergencies = int(sum(result.emergency))

    # tham so ly thuyet M/G/1 tong hop (dung de doi chieu / kiem chung)
    total_demand = cfg.arrival_rate + cfg.departure_rate
    if total_demand > 0:
        w_arr = cfg.arrival_rate / total_demand
        w_dep = cfg.departure_rate / total_demand
    else:
        w_arr = w_dep = 0.0
    mean_service_sec = w_arr * cfg.service_arr + w_dep * cfg.service_dep
    mu_theoretical = 3600.0 / mean_service_sec if mean_service_sec > 0 else 0.0
    var_sec2 = (
        w_arr * (cfg.service_arr - mean_service_sec) ** 2
        + w_dep * (cfg.service_dep - mean_service_sec) ** 2
    )
    sigma_hour = math.sqrt(var_sec2) / 3600.0

    theory = pk_theoretical(total_demand, mu_theoretical, sigma_hour)

    result.summary = {
        "n_cycles": n,
        "cum_arrived": cum_arrived,
        "cum_dep_demand": cum_dep_demand,
        "cum_served": cum_served,
        "served_arr_total": int(sum(result.served_arr)),
        "served_dep_total": int(sum(result.served_dep)),
        "avg_lq_arr": avg_lq_arr,
        "avg_lq_dep": avg_lq_dep,
        "avg_lq_total": avg_lq_total,
        "avg_utilization": avg_util,
        "wq_arr_min": wq_arr_min,
        "wq_dep_min": wq_dep_min,
        "wq_total_min": wq_total_min,
        "throughput_per_h": throughput_per_h,
        "n_emergencies": n_emergencies,
        "mu_theoretical": mu_theoretical,
        "sigma_hour": sigma_hour,
        "theory_rho": theory["rho"],
        "theory_Lq": theory["Lq"],
        "theory_Wq_min": theory["Wq_min"],
    }

    return result
