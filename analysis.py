# -*- coding: utf-8 -*-
"""
따릉이 '언덕 이동격차' 분석
- 경사 보정 자전거 필요동력 지수(Hill Power Index, HPI)로 대여소별 '도착 난이도'를 계산하고
- 대여·반납 불균형, 오르막/내리막 방향 비대칭을 분석한 뒤
- 머신러닝으로 경사의 설명력을 검증하고, 전기자전거 우선배치 대여소와 탄소감축 시나리오를 산출한다.

실행 예)
  python analysis.py --stations "data/공공자전거 대여소 정보(26.6월 기준).xlsx" --trips "data/trips/*.csv"
  python analysis.py --demo            # 가짜 데이터로 코드 작동만 확인 (결과를 보고서에 쓰면 안 됨)

출력: outputs/ 폴더 (그림 PNG, 결과 CSV, results_summary.txt, results.json)
"""
import argparse
import glob
import json
import math
import os
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_score

# ---------------------------------------------------------------------------
# 0. 물리 파라미터 (자전거 주행 동력 모델, Martin et al., 1998)
#    P = v * m g (sinθ + Crr cosθ) + 0.5 ρ CdA v^3
# ---------------------------------------------------------------------------
M_RIDER = 70.0      # 탑승자 질량 [kg] (가정)
M_BIKE = 18.0       # 따릉이 일반형 질량 [kg] (약 18kg)
G = 9.81            # 중력가속도 [m/s^2]
CRR = 0.008         # 구름저항계수 (도심 일반 자전거 가정)
CDA = 0.6           # 항력계수×전면면적 [m^2] (직립 자세 가정)
RHO = 1.2           # 공기밀도 [kg/m^3]
V_REF = 15 / 3.6    # 기준 주행속도 15 km/h [m/s]
CIRCUITY = 1.3      # 직선거리 → 실제 경로거리 보정계수 (가정)
P_EASY = 100.0      # '편안한' 지속 출력 상한 [W] (가정)
P_HARD = 150.0      # 일반 시민에게 '부담스러운' 출력 기준 [W] (가정)
MIN_PAIR_DIST = 300  # 경사 계산 최소 직선거리 [m] (너무 가까우면 고도오차가 경사를 왜곡)

# 탄소 시나리오 파라미터 (보고서에 출처와 함께 기재할 것)
CAR_SUBSTITUTION = [0.05, 0.10, 0.20]  # 새로 생긴 자전거 이용 중 승용차를 대체하는 비율 (시나리오)
CAR_EF_KG_PER_KM = 0.13                # 중형 승용차 1인·km당 CO2 [kg] (World Watch Institute 계수, 그린포스트코리아 2020.6.7 인용)

OUT = "outputs"


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def set_korean_font():
    for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def read_csv_any(path, **kw):
    """공공데이터 CSV는 utf-8 / cp949가 섞여 있으므로 순서대로 시도"""
    last = None
    for enc in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc, **kw)
        except (UnicodeDecodeError, UnicodeError) as e:
            last = e
    raise last


def norm_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = s.lstrip("0")
    return s if s else "0"


def find_col(cols, must, exclude=()):
    for c in cols:
        k = str(c).replace(" ", "")
        if all(m in k for m in must) and not any(e in k for e in exclude):
            return c
    return None


def haversine(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def required_power(grade, v=V_REF):
    """경사(grade=Δh/거리)에서 일정 속도 v로 오를 때 필요한 사람 출력 [W]. 내리막에서 음수면 0(관성 주행)."""
    theta = np.arctan(grade)
    m = M_RIDER + M_BIKE
    p = v * m * G * (np.sin(theta) + CRR * np.cos(theta)) + 0.5 * RHO * CDA * v ** 3
    return np.maximum(p, 0.0)


# ---------------------------------------------------------------------------
# 1. 데이터 불러오기
# ---------------------------------------------------------------------------
def load_stations(path):
    """서울시 공공자전거 대여소 정보 (xlsx/csv). 위쪽 여러 줄에 나뉜 헤더를 합쳐서 처리한다."""
    if str(path).lower().endswith((".xlsx", ".xls")):   # 최근 파일은 엑셀(.xlsx)로 제공됨
        raw = pd.read_excel(path, header=None, dtype=str)
    else:
        raw = read_csv_any(path, header=None, dtype=str)
    raw = raw.apply(lambda col: col.map(lambda v: None if pd.isna(v) else str(v).replace("\n", "").strip()))
    # 첫 데이터 행 = 서울 위도·경도 값이 있는 첫 행
    def is_data(row):
        nums = pd.to_numeric(row, errors="coerce")
        return nums.between(37.4, 37.72).any() and nums.between(126.7, 127.25).any()
    first = next(i for i in range(len(raw)) if is_data(raw.iloc[i]))
    hdr_start = next(i for i in range(first) if raw.iloc[i].fillna("").str.contains("대여소").any())
    hdr = raw.iloc[hdr_start:first].fillna("")
    cols = ["".join(hdr[c].tolist()).replace(" ", "") or f"col{c}" for c in hdr.columns]
    df = raw.iloc[first:].copy()
    df.columns = cols
    c_id = find_col(cols, ["대여소", "번호"])
    c_name = find_col(cols, ["보관소"]) or find_col(cols, ["대여소명"]) or find_col(cols, ["명"], exclude=["번호"])
    c_gu = find_col(cols, ["자치구"])
    c_lat = find_col(cols, ["위도"])
    c_lon = find_col(cols, ["경도"])
    dock_cols = [c for c in cols if "거치" in c and "운영" not in c]
    st = pd.DataFrame({
        "station_id": df[c_id].map(norm_id),
        "name": df[c_name] if c_name else df[c_id],
        "gu": df[c_gu] if c_gu else "미상",
        "lat": pd.to_numeric(df[c_lat], errors="coerce"),
        "lon": pd.to_numeric(df[c_lon], errors="coerce"),
    })
    docks = np.zeros(len(df))
    for c in dict.fromkeys(dock_cols):
        docks += pd.to_numeric(df[c], errors="coerce").fillna(0).values
    st["docks"] = docks
    st = st[(st.lat.between(37.40, 37.72)) & (st.lon.between(126.75, 127.20))]
    st = st.dropna(subset=["station_id"]).drop_duplicates("station_id").reset_index(drop=True)
    return st


def load_trips(pattern, chunksize=500_000):
    """서울특별시 공공자전거 대여이력 정보 CSV(여러 개 가능) → O-D 집계"""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"대여이력 파일을 찾지 못했습니다: {pattern}")
    od_parts, dates, n_total, dist_sum, dist_n = [], set(), 0, 0.0, 0
    for f in files:
        head = read_csv_any(f, nrows=5)
        cols = list(head.columns)
        c_o = find_col(cols, ["대여소번호"], exclude=["반납"])
        c_d = find_col(cols, ["반납", "대여소번호"])
        c_t = find_col(cols, ["대여일시"]) or find_col(cols, ["대여", "일"], exclude=["대여소"])
        c_dist = find_col(cols, ["이용거리"])
        if c_o is None or c_d is None:
            raise ValueError(f"{f}: 대여/반납 대여소번호 컬럼을 찾지 못했습니다. 컬럼: {cols}")
        use = [c for c in [c_o, c_d, c_t, c_dist] if c]
        enc = None
        for e in ["utf-8-sig", "cp949", "euc-kr"]:
            try:
                pd.read_csv(f, encoding=e, nrows=5)
                enc = e
                break
            except (UnicodeDecodeError, UnicodeError):
                pass
        print(f"  - {os.path.basename(f)} 읽는 중 ...")
        for ch in pd.read_csv(f, encoding=enc, usecols=use, dtype=str, chunksize=chunksize):
            ch = ch.dropna(subset=[c_o, c_d])
            o = ch[c_o].map(norm_id)
            d = ch[c_d].map(norm_id)
            n_total += len(ch)
            if c_t:
                dates.update(ch[c_t].str[:10].dropna().unique())
            if c_dist:
                dist = pd.to_numeric(ch[c_dist], errors="coerce")
                ok = dist.between(100, 50_000)       # 100m~50km만 유효
                dist_sum += dist[ok].sum()
                dist_n += int(ok.sum())
            od_parts.append(pd.DataFrame({"o": o, "d": d}).value_counts().rename("n").reset_index())
    od = pd.concat(od_parts).groupby(["o", "d"], as_index=False)["n"].sum()
    n_days = max(len(dates), 1)
    avg_trip_m = dist_sum / dist_n if dist_n else np.nan
    return od, n_days, n_total, avg_trip_m


def save_od(od, n_days, n_total, avg_trip_m, path):
    """대여이력 원본(수백 MB)을 출발–도착 대여소별 통행 수(수 MB)로 줄여 저장 → GitHub에 함께 올려 재현 가능"""
    od.to_csv(path, index=False, compression="gzip" if str(path).endswith(".gz") else None)
    meta = {"n_days": int(n_days), "n_total": int(n_total), "avg_trip_m": None if np.isnan(avg_trip_m) else float(avg_trip_m)}
    with open(str(path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_od(path):
    od = pd.read_csv(path, dtype={"o": str, "d": str})
    with open(str(path) + ".meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    avg = meta["avg_trip_m"] if meta["avg_trip_m"] is not None else np.nan
    return od, meta["n_days"], meta["n_total"], avg


def get_elevations(st, cache_path, dataset="srtm30m", elev_csv=None):
    """대여소 고도: (1) 사용자가 준 CSV(station_id, elev) → (2) 캐시 → (3) Open Topo Data API"""
    if elev_csv:
        e = read_csv_any(elev_csv, dtype={"station_id": str})
        e["station_id"] = e["station_id"].map(norm_id)
        return st.merge(e[["station_id", "elev"]], on="station_id", how="left")
    cache = pd.DataFrame(columns=["station_id", "elev"])
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path, dtype={"station_id": str})
    need = st[~st.station_id.isin(cache.station_id)]
    if len(need):
        import requests
        print(f"  - 고도 조회: {len(need)}개 대여소 (Open Topo Data {dataset}, 100개씩/1초 간격)")
        rows = []
        for i in range(0, len(need), 100):
            b = need.iloc[i:i + 100]
            locs = "|".join(f"{a:.6f},{o:.6f}" for a, o in zip(b.lat, b.lon))
            for attempt in range(3):
                try:
                    r = requests.get(f"https://api.opentopodata.org/v1/{dataset}",
                                     params={"locations": locs}, timeout=30)
                    r.raise_for_status()
                    res = r.json()["results"]
                    rows += [{"station_id": sid, "elev": x["elevation"]} for sid, x in zip(b.station_id, res)]
                    break
                except Exception as ex:  # 네트워크 오류 시 재시도
                    print("    재시도:", ex)
                    time.sleep(3)
            time.sleep(1.1)
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
    return st.merge(cache, on="station_id", how="left")


# ---------------------------------------------------------------------------
# 2. 대여소별 지형·동력 특성 (이용 데이터와 무관한 '순수 지형' 변수 → 누설 방지)
# ---------------------------------------------------------------------------
def station_terrain_features(st):
    lat, lon, z = st.lat.values, st.lon.values, st.elev.values
    n = len(st)
    feats = {k: np.full(n, np.nan) for k in
             ["rel_elev_500", "rel_elev_1000", "hpi_in", "hpi_out", "hard_in_share", "density_500"]}
    for i in range(n):
        d0 = haversine(lat[i], lon[i], lat, lon)
        d0[i] = np.nan
        for r in (500, 1000):
            m = d0 <= r
            if m.sum() >= 3:
                feats[f"rel_elev_{r}"][i] = z[i] - np.nanmean(z[m])
        feats["density_500"][i] = np.nansum(d0 <= 500)
        m = (d0 >= MIN_PAIR_DIST) & (d0 <= 2000)          # 자전거 단거리 통행권(0.3~2km) 이웃
        if m.sum() >= 3:
            dist = d0[m] * CIRCUITY
            g_in = (z[i] - z[m]) / dist                     # 이웃 → 이 대여소로 올 때 경사
            p_in = required_power(np.clip(g_in, -0.15, 0.15))
            p_out = required_power(np.clip(-g_in, -0.15, 0.15))
            feats["hpi_in"][i] = p_in.mean()
            feats["hpi_out"][i] = p_out.mean()
            feats["hard_in_share"][i] = (p_in > P_HARD).mean()
    for k, v in feats.items():
        st[k] = v
    return st


# ---------------------------------------------------------------------------
# 3. 분석
# ---------------------------------------------------------------------------
def analyze(st, od, n_days, n_total, avg_trip_m, min_station_trips=100, min_pair_trips=10):
    res = {"n_trips": int(n_total), "n_days": int(n_days), "n_stations_all": int(len(st))}
    idx = st.set_index("station_id")

    # 3-1 대여소별 대여·반납 불균형
    rent = od.groupby("o")["n"].sum()
    ret = od.groupby("d")["n"].sum()
    st["rent"] = st.station_id.map(rent).fillna(0)
    st["ret"] = st.station_id.map(ret).fillna(0)
    st["total"] = st.rent + st.ret
    st["imbalance"] = (st.ret - st.rent) / st.total.replace(0, np.nan)   # 음수 = 자전거 유출(부족)
    st["drain_per_day"] = (st.rent - st.ret) / n_days                    # 양수 = 하루 평균 순유출 대수
    st["daily_use"] = st.total / n_days
    sa = st.dropna(subset=["elev", "hpi_in", "rel_elev_1000", "imbalance"])
    sa = sa[sa.total >= min_station_trips].copy()
    res["n_stations_used"] = int(len(sa))
    for f in ["rel_elev_1000", "hpi_in", "elev"]:
        rho, p = spearmanr(sa[f], sa.imbalance)
        res[f"spearman_{f}_vs_imbalance"] = {"rho": round(float(rho), 3), "p": float(p)}
    # 언덕 대여소(도착 동력 상위 20%) vs 평지(하위 20%)
    q80, q20 = sa.hpi_in.quantile(0.8), sa.hpi_in.quantile(0.2)
    hill, flat = sa[sa.hpi_in >= q80], sa[sa.hpi_in <= q20]
    res["hill_vs_flat"] = {
        "hpi_cut_hill_W": round(float(q80), 1), "hpi_cut_flat_W": round(float(q20), 1),
        "hill_mean_imbalance": round(float(hill.imbalance.mean()), 3),
        "flat_mean_imbalance": round(float(flat.imbalance.mean()), 3),
        "hill_drain_per_day_sum": round(float(hill.drain_per_day.clip(lower=0).sum()), 1),
        "hill_mean_daily_use": round(float(hill.daily_use.mean()), 1),
        "flat_mean_daily_use": round(float(flat.daily_use.mean()), 1),
    }
    res["all_drain_per_day_sum"] = round(float(sa.drain_per_day.clip(lower=0).sum()), 1)

    # 3-2 같은 두 대여소 사이 오르막/내리막 방향 비대칭 (같은 구간이라 수요 조건이 통제됨)
    od2 = od[od.o != od.d].copy()
    od2 = od2[od2.o.isin(idx.index) & od2.d.isin(idx.index)]
    a = idx.loc[od2.o, ["lat", "lon", "elev"]].values
    b = idx.loc[od2.d, ["lat", "lon", "elev"]].values
    od2["d0"] = haversine(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    od2["grade"] = (b[:, 2] - a[:, 2]) / (od2.d0 * CIRCUITY)
    od2 = od2[(od2.d0 >= MIN_PAIR_DIST) & od2.grade.notna()]
    od2["power"] = required_power(np.clip(od2.grade, -0.15, 0.15))
    od2["key"] = [tuple(sorted(x)) for x in zip(od2.o, od2.d)]
    od2["down"] = od2.grade < 0
    pair = od2.groupby("key").agg(n=("n", "sum"), g=("grade", lambda s: s.abs().max()))
    down_n = od2[od2.down].groupby("key")["n"].sum()
    pair["n_down"] = down_n.reindex(pair.index).fillna(0)
    pair = pair[pair.n >= min_pair_trips]
    pair["down_share"] = pair.n_down / pair.n
    bins = [0, 0.01, 0.02, 0.03, 0.05, 1]
    labels = ["0~1%", "1~2%", "2~3%", "3~5%", "5% 이상"]
    pair["gbin"] = pd.cut(pair.g, bins, labels=labels, include_lowest=True)
    tab = pair.groupby("gbin", observed=False).apply(
        lambda x: pd.Series({"pairs": len(x), "trips": x.n.sum(),
                             "down_share": (x.n_down.sum() / x.n.sum()) if x.n.sum() else np.nan}))
    res["asymmetry_by_grade"] = {str(k): {"pairs": int(v.pairs), "trips": int(v.trips),
                                          "down_share_pct": round(float(v.down_share) * 100, 1)}
                                 for k, v in tab.iterrows()}
    # 로지스틱 회귀: 경사가 1%p 커질 때 '내리막 방향으로 이용할' 오즈 변화
    X = np.r_[pair.g.values, pair.g.values].reshape(-1, 1) * 100     # % 단위
    y = np.r_[np.ones(len(pair)), np.zeros(len(pair))]
    w = np.r_[pair.n_down.values, (pair.n - pair.n_down).values]
    lr = LogisticRegression(penalty=None).fit(X, y, sample_weight=w)
    res["logit_odds_ratio_per_1pct_grade"] = round(float(np.exp(lr.coef_[0][0])), 3)
    # 실제 통행의 필요 출력 분포
    tw = od2.n.sum()
    res["trip_power_share"] = {
        "<=100W": round(float(od2.loc[od2.power <= P_EASY, "n"].sum() / tw * 100), 1),
        "100~150W": round(float(od2.loc[(od2.power > P_EASY) & (od2.power <= P_HARD), "n"].sum() / tw * 100), 1),
        ">150W": round(float(od2.loc[od2.power > P_HARD, "n"].sum() / tw * 100), 1),
    }
    od_km = float((od2.d0 * CIRCUITY * od2.n).sum() / tw / 1000)              # 출발≠도착 통행의 경로거리 추정
    all_km = float(avg_trip_m / 1000) if not np.isnan(avg_trip_m) else od_km      # 이용거리(왕복·여가 포함)
    res["avg_trip_km_all"] = round(all_km, 2)
    res["avg_trip_km_od"] = round(od_km, 2)
    res["avg_trip_km"] = round(min(all_km, od_km), 2)                             # 탄소 계산은 보수적인 값 사용

    # 3-3 머신러닝: 경사 변수가 불균형·이용량을 얼마나 더 설명하는가 (Random Forest, 5-fold CV)
    base_f = ["density_500", "docks"]
    hill_f = ["elev", "rel_elev_500", "rel_elev_1000", "hpi_in", "hpi_out", "hard_in_share"]
    ml = sa.copy()
    for f in base_f + hill_f:          # 주변 대여소가 적어 계산이 안 된 값은 중앙값으로 대체
        ml[f] = ml[f].fillna(ml[f].median())
    kf = KFold(5, shuffle=True, random_state=42)
    rf = lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=5, random_state=42, n_jobs=-1)
    out, out_sp = {}, {}
    gkf = GroupKFold(5)                                   # 공간 교차검증: 같은 자치구는 학습/평가에 섞이지 않게
    groups = ml.gu.values
    for tgt in ["imbalance", "log_use"]:
        yv = ml.imbalance.values if tgt == "imbalance" else np.log1p(ml.daily_use.values)
        r2_base = cross_val_score(rf(), ml[base_f], yv, cv=kf, scoring="r2").mean()
        r2_full = cross_val_score(rf(), ml[base_f + hill_f], yv, cv=kf, scoring="r2").mean()
        out[tgt] = {"r2_base": round(float(r2_base), 3), "r2_with_hill": round(float(r2_full), 3)}
        s_base = cross_val_score(rf(), ml[base_f], yv, cv=gkf, groups=groups, scoring="r2").mean()
        s_full = cross_val_score(rf(), ml[base_f + hill_f], yv, cv=gkf, groups=groups, scoring="r2").mean()
        out_sp[tgt] = {"r2_base": round(float(s_base), 3), "r2_with_hill": round(float(s_full), 3)}
    res["ml_r2"] = out
    res["ml_r2_spatial_cv"] = out_sp
    # 순열 중요도는 학습에 쓰지 않은 검증 fold에서 계산(과대평가 방지)
    feats = base_f + hill_f
    imps = []
    for tr, te in kf.split(ml):
        m_ = rf().fit(ml.iloc[tr][feats], ml.imbalance.iloc[tr])
        pi = permutation_importance(m_, ml.iloc[te][feats], ml.imbalance.iloc[te], n_repeats=5, random_state=42)
        imps.append(pi.importances_mean)
    imp = pd.Series(np.mean(imps, axis=0), index=feats).sort_values(ascending=False)
    res["perm_importance_imbalance"] = {k: round(float(v), 4) for k, v in imp.items()}

    # 3-4 반사실 추정: '평지만큼 오르기 쉬워지면'(전기자전거 보조) 이용량이 얼마나 늘까
    m_use = rf().fit(ml[base_f + hill_f], np.log1p(ml.daily_use))
    flat_ref = ml[ml.hpi_in <= ml.hpi_in.quantile(0.2)]
    cf = ml[base_f + hill_f].copy()
    for f in hill_f:
        if f == "elev":
            continue
        cf[f] = np.minimum(cf[f], flat_ref[f].median()) if f.startswith(("hpi", "hard")) else np.minimum(cf[f], 0.0)
    ml["use_pred"] = np.expm1(m_use.predict(ml[base_f + hill_f]))
    ml["use_cf"] = np.expm1(m_use.predict(cf))
    ml["latent_gain"] = (ml.use_cf - ml.use_pred).clip(lower=0)      # 하루 잠재 추가 이용(대여+반납)

    # 3-5 전기자전거 우선배치 점수 = 도착난이도(40%) + 순유출(30%) + 잠재수요(30%) [백분위 정규화]
    pct = lambda s: s.rank(pct=True)
    ml["priority"] = 0.4 * pct(ml.hpi_in) + 0.3 * pct(ml.drain_per_day) + 0.3 * pct(ml.latent_gain)
    top = ml.sort_values("priority", ascending=False).head(30)
    res["top30_latent_gain_per_day"] = round(float(top.latent_gain.sum()), 1)
    res["top30_drain_per_day"] = round(float(top.drain_per_day.clip(lower=0).sum()), 1)
    res["top30_gu_counts"] = top.gu.value_counts().to_dict()

    # 3-6 탄소 시나리오: 잠재 추가 '통행' = 잠재 추가(대여+반납)/2
    trips_year = res["top30_latent_gain_per_day"] / 2 * 365
    res["carbon_scenarios_tCO2_per_year"] = {
        f"대체율{int(s*100)}%": round(trips_year * res["avg_trip_km"] * s * CAR_EF_KG_PER_KM / 1000, 1)
        for s in CAR_SUBSTITUTION}
    res["latent_trips_per_year_top30"] = round(trips_year)

    # 3-7 자치구별 형평성 요약
    gu = sa.groupby("gu").agg(stations=("station_id", "count"), hpi_in=("hpi_in", "mean"),
                              hill_station_share=("hpi_in", lambda s: (s >= q80).mean() * 100),   # 서울 상위20% 언덕 대여소 비율
                              hard_route_share=("hard_in_share", lambda s: s.mean() * 100),     # 접근 경로 중 150W 초과 비율
                              imbalance=("imbalance", "mean"), daily_use=("daily_use", "mean"))
    gu = gu.sort_values("hpi_in", ascending=False).round(2)
    return st, sa, ml, pair, tab, imp, top, gu, res


# ---------------------------------------------------------------------------
# 3-8 강건성 검증: (1) 자치구 단위 군집 부트스트랩 신뢰구간 (2) 물리 가정 민감도
# ---------------------------------------------------------------------------
def hpi_in_only(st, v_kmh=15, m_rider=M_RIDER, circuity=CIRCUITY, rmax=2000, crr=CRR):
    lat, lon, z = st.lat.values, st.lon.values, st.elev.values
    v = v_kmh / 3.6
    m = m_rider + M_BIKE
    out = np.full(len(st), np.nan)
    for i in range(len(st)):
        d0 = haversine(lat[i], lon[i], lat, lon); d0[i] = np.nan
        k = (d0 >= MIN_PAIR_DIST) & (d0 <= rmax)
        if k.sum() >= 3:
            g = np.clip((z[i] - z[k]) / (d0[k] * circuity), -0.15, 0.15)
            th = np.arctan(g)
            p = v * m * G * (np.sin(th) + crr * np.cos(th)) + 0.5 * RHO * CDA * v ** 3
            out[i] = np.maximum(p, 0).mean()
    return out


def robustness(sa, st_all=None, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    gus = sa.gu.unique()
    by_gu = {g: sa.index[sa.gu == g].values for g in gus}
    rows = {"hpi": [], "elev": [], "rel": [], "diff": []}
    for _ in range(n_boot):
        pick = np.concatenate([by_gu[g] for g in rng.choice(gus, len(gus), replace=True)])
        s = sa.loc[pick]
        r_h = spearmanr(s.hpi_in, s.imbalance)[0]; r_e = spearmanr(s.elev, s.imbalance)[0]
        r_r = spearmanr(s.rel_elev_1000, s.imbalance)[0]
        rows["hpi"].append(r_h); rows["elev"].append(r_e); rows["rel"].append(r_r); rows["diff"].append(r_h - r_e)
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    res["cluster_boot_ci95"] = {k: ci(v) for k, v in rows.items()}
    # 물리 가정 민감도: 가정을 바꿔 HPI를 다시 계산해도 결론(상관)이 유지되는가
    settings = {"기준(15km/h, 70kg, 우회 1.3, 반경 2km)": {},
                "속도 10km/h": {"v_kmh": 10}, "속도 20km/h": {"v_kmh": 20},
                "탑승자 55kg": {"m_rider": 55}, "탑승자 85kg": {"m_rider": 85},
                "우회계수 1.2": {"circuity": 1.2}, "우회계수 1.4": {"circuity": 1.4},
                "이웃 반경 1.5km": {"rmax": 1500}, "이웃 반경 3km": {"rmax": 3000},
                "구름저항 0.006": {"crr": 0.006}, "구름저항 0.012": {"crr": 0.012}}
    sens = {}
    base = None
    ref = st_all if st_all is not None else sa
    for name, kw in settings.items():
        h_all = pd.Series(hpi_in_only(ref, **kw), index=ref.station_id.values)   # 이웃은 전체 대여소 기준
        h = h_all.reindex(sa.station_id.values).values
        ok = ~np.isnan(h)
        rho = spearmanr(h[ok], sa.imbalance.values[ok])[0]
        if base is None:
            base = h
        rank_agree = spearmanr(h[ok], base[ok])[0]
        top20 = (h >= np.nanpercentile(h, 80)); top20b = (base >= np.nanpercentile(base, 80))
        overlap = (top20 & top20b).sum() / top20b.sum()
        sens[name] = {"rho_vs_imbalance": round(float(rho), 3), "rank_corr_with_base": round(float(rank_agree), 3),
                      "hill_top20_overlap_pct": round(float(overlap) * 100, 1)}
    res["sensitivity"] = sens
    return res


# ---------------------------------------------------------------------------
# 4. 시각화
# ---------------------------------------------------------------------------
def figures(sa, ml, tab, imp, top, gu, res, demo=False):
    set_korean_font()
    wm = (lambda ax: ax.text(0.5, 0.5, "DEMO DATA\n(가짜 데이터)", transform=ax.transAxes, fontsize=30,
                              color="red", alpha=0.25, ha="center", va="center", rotation=20)) if demo else (lambda ax: None)

    # 그림1 경사별 필요 출력 (물리 모델)
    fig, ax = plt.subplots(figsize=(7, 4))
    g = np.linspace(0, 0.10, 101)
    for v_kmh in (10, 15, 20):
        ax.plot(g * 100, required_power(g, v_kmh / 3.6), label=f"{v_kmh} km/h")
    ax.axhline(P_EASY, ls="--", c="gray"); ax.text(7.0, P_EASY - 22, f"편안한 출력 {P_EASY:.0f} W", color="gray")
    ax.axhline(P_HARD, ls="--", c="crimson"); ax.text(7.0, P_HARD - 22, f"부담 기준 {P_HARD:.0f} W", color="crimson")
    ax.set_xlabel("경사 (%)"); ax.set_ylabel("필요 출력 (W)")
    ax.set_title("따릉이(탑승자 70kg + 자전거 18kg) 경사별 필요 출력")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_power_curve.png", dpi=200); plt.close(fig)

    # 그림2 대여소 지도: 도착 난이도 / 불균형
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    s1 = axs[0].scatter(sa.lon, sa.lat, c=sa.hpi_in, s=6, cmap="viridis",
                        vmin=np.nanpercentile(sa.hpi_in, 5), vmax=np.nanpercentile(sa.hpi_in, 95))
    fig.colorbar(s1, ax=axs[0], label="도착 필요 출력 HPI (W)"); axs[0].set_title("대여소별 '도착 난이도'(HPI)")
    lim = np.nanpercentile(np.abs(sa.imbalance), 95)
    s2 = axs[1].scatter(sa.lon, sa.lat, c=sa.imbalance, s=6, cmap="RdBu", vmin=-lim, vmax=lim)
    fig.colorbar(s2, ax=axs[1], label="불균형 지수 (−: 자전거 부족)"); axs[1].set_title("대여소별 대여·반납 불균형")
    for ax in axs:
        ax.set_xlabel("경도"); ax.set_ylabel("위도"); ax.set_aspect(1.25); wm(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_station_maps.png", dpi=200); plt.close(fig)

    # 그림3 경사 구간별 내리막 방향 이용 비율
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = tab.down_share.values * 100
    bars = ax.bar(tab.index.astype(str), vals, color="#3b6fb6")
    ax.axhline(50, ls="--", c="gray"); ax.text(4.45, 50.6, "대칭(50%)", color="gray", ha="right")
    for b_, v_, n_ in zip(bars, vals, tab.pairs.values):
        if not np.isnan(v_):
            ax.text(b_.get_x() + b_.get_width() / 2, v_ + 0.8, f"{v_:.1f}%\n(n={int(n_)})", ha="center", fontsize=8)
    ax.set_ylim(40, max(np.nanmax(vals) + 8, 60))
    ax.set_xlabel("두 대여소 사이 경사"); ax.set_ylabel("내리막 방향 이용 비율 (%)")
    ax.set_title("같은 구간이라도 경사가 클수록 내리막 방향 이용이 많다"); wm(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_downhill_asymmetry.png", dpi=200); plt.close(fig)

    # 그림4 ML 성능 비교 + 변수 중요도
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
    r2 = res["ml_r2"]
    names = ["불균형 지수", "이용량(log)"]
    xb = np.arange(2)
    sp = res.get("ml_r2_spatial_cv", r2)
    w_ = 0.26
    bars = [(-w_, [r2["imbalance"]["r2_base"], r2["log_use"]["r2_base"]], "기본 변수만", "#bbbbbb"),
            (0, [r2["imbalance"]["r2_with_hill"], r2["log_use"]["r2_with_hill"]], "+ 경사·동력 변수", "#d1495b"),
            (w_, [sp["imbalance"]["r2_with_hill"], sp["log_use"]["r2_with_hill"]], "+ 경사·동력 (자치구 단위 검증)", "#7a1f2b")]
    for off, vals, lab, col in bars:
        bb = axs[0].bar(xb + off, vals, w_, label=lab, color=col)
        for rect, v_ in zip(bb, vals):
            axs[0].text(rect.get_x() + rect.get_width() / 2, max(v_, 0) + 0.01, f"{v_:.2f}", ha="center", fontsize=8)
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_xticks(xb); axs[0].set_xticklabels(names); axs[0].set_ylabel("교차검증 R²")
    axs[0].set_title("Random Forest 교차검증 R²"); axs[0].legend(fontsize=8, loc="upper right")
    kor = {"density_500": "주변 대여소 밀도", "docks": "거치대 수", "elev": "해발고도", "rel_elev_500": "상대고도(500m)",
           "rel_elev_1000": "상대고도(1km)", "hpi_in": "도착 필요출력", "hpi_out": "출발 필요출력", "hard_in_share": "150W 초과 접근 비율"}
    imp_s = imp.sort_values()
    axs[1].barh([kor.get(i, i) for i in imp_s.index], imp_s.values, color="#3b6fb6")
    axs[1].set_title("불균형 예측 변수 중요도 (검증 데이터 순열 중요도)")
    for ax in axs:
        wm(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_ml.png", dpi=200); plt.close(fig)

    # 그림5 전기자전거 우선배치 대여소 + 자치구 형평성
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    axs[0].scatter(ml.lon, ml.lat, s=4, c="#cccccc")
    axs[0].scatter(top.lon, top.lat, s=40, c="crimson", edgecolor="k", lw=.5, label="우선배치 상위 30")
    for r_, (x_, y_) in enumerate(zip(top.lon, top.lat), 1):
        if r_ <= 10:
            axs[0].text(x_, y_, str(r_), fontsize=8)
    axs[0].set_aspect(1.25); axs[0].set_title("전기자전거 우선배치 대여소"); axs[0].legend(loc="lower right")
    gg = gu.sort_values("hill_station_share").tail(15)
    axs[1].barh(gg.index, gg.hill_station_share, color="#e07a1f")
    axs[1].set_xlabel("서울 HPI 상위 20% '언덕 대여소' 비율 (%)"); axs[1].set_title("자치구별 언덕 대여소 비율 (상위 15)")
    for ax in axs:
        wm(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_priority_equity.png", dpi=200); plt.close(fig)


def write_summary(res, top, gu):
    hv = res["hill_vs_flat"]
    sp = res["spearman_hpi_in_vs_imbalance"]
    asym = res["asymmetry_by_grade"]
    lines = [
        "===== 보고서 붙여넣기용 결과 요약 =====",
        f"분석 통행: {res['n_trips']:,}건 / {res['n_days']}일, 분석 대여소: {res['n_stations_used']:,}개 (전체 {res['n_stations_all']:,}개 중 이용 100건 이상)",
        f"[결과1] 도착 필요출력(HPI)과 불균형 지수의 스피어만 상관: ρ = {sp['rho']} (p = {sp['p']:.2e})",
        f"[결과1] 언덕 대여소(HPI 상위20%, ≥{hv['hpi_cut_hill_W']}W) 평균 불균형 {hv['hill_mean_imbalance']} vs 평지(하위20%, ≤{hv['hpi_cut_flat_W']}W) {hv['flat_mean_imbalance']}",
        f"[결과1] 언덕 대여소 하루 순유출 합계: {hv['hill_drain_per_day_sum']}대 (전체 유출 대여소 합계 {res['all_drain_per_day_sum']}대)",
        f"[결과1] 하루 평균 이용(대여+반납): 언덕 {hv['hill_mean_daily_use']}건 vs 평지 {hv['flat_mean_daily_use']}건",
        "[결과2] 경사 구간별 내리막 방향 이용 비율: " + ", ".join(f"{k} {v['down_share_pct']}% (구간 {v['pairs']}개)" for k, v in asym.items()),
        f"[결과2] 로지스틱 회귀: 경사 1%p 증가 시 내리막 방향 이용 오즈 {res['logit_odds_ratio_per_1pct_grade']}배",
        f"[결과2] 실제 통행 필요출력 분포: ≤100W {res['trip_power_share']['<=100W']}%, 100~150W {res['trip_power_share']['100~150W']}%, >150W {res['trip_power_share']['>150W']}%",
        f"[결과3] 불균형 예측 R²: 기본 {res['ml_r2']['imbalance']['r2_base']} → 경사 추가 {res['ml_r2']['imbalance']['r2_with_hill']}",
        f"[결과3] 이용량 예측 R²: 기본 {res['ml_r2']['log_use']['r2_base']} → 경사 추가 {res['ml_r2']['log_use']['r2_with_hill']}",
        f"[결과3] 자치구 단위 공간 교차검증 R²: 불균형 {res['ml_r2_spatial_cv']['imbalance']}, 이용량 {res['ml_r2_spatial_cv']['log_use']}",
        "[결과3] 변수 중요도 상위 3: " + ", ".join(list(res["perm_importance_imbalance"].keys())[:3]),
        f"[결과4] 우선배치 상위30 대여소: 잠재 추가 이용 {res['top30_latent_gain_per_day']}건/일, 현재 순유출 {res['top30_drain_per_day']}대/일",
        f"[결과4] 상위30 자치구 분포: {res['top30_gu_counts']}",
        f"[결과4] 평균 통행거리 {res['avg_trip_km']}km, 잠재 추가 통행 {res['latent_trips_per_year_top30']:,}건/년",
        f"[결과4] 탄소감축 시나리오(tCO2/년, 승용차 {CAR_EF_KG_PER_KM}kg/km 가정): {res['carbon_scenarios_tCO2_per_year']}",
        f"[강건성] 자치구 군집 부트스트랩 95% CI: {res.get('robustness', {}).get('cluster_boot_ci95')}",
        f"[강건성] 물리 가정 민감도: {res.get('robustness', {}).get('sensitivity')}",
        f"[참고] 평균 통행거리: 이용거리 기준 {res['avg_trip_km_all']}km, 편도 O-D 기준 {res['avg_trip_km_od']}km (탄소 계산은 작은 값 사용)",
        "",
        "자치구별 요약(도착 난이도 순):",
        gu.to_string(),
        "",
        "우선배치 상위 10:",
        top.head(10)[["name", "gu", "elev", "hpi_in", "imbalance", "drain_per_day", "latent_gain", "priority"]].round(2).to_string(index=False),
    ]
    txt = "\n".join(lines)
    with open(f"{OUT}/results_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)


# ---------------------------------------------------------------------------
# 5. DEMO용 가짜 데이터 (코드 작동 확인 전용!)
# ---------------------------------------------------------------------------
def make_demo(n_st=1500, n_trips=400_000, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(37.45, 37.68, n_st); lon = rng.uniform(126.80, 127.15, n_st)
    hills = [(37.55, 126.98, 120, 0.02), (37.60, 127.05, 80, 0.015), (37.48, 126.95, 100, 0.02)]
    elev = 15 + sum(h * np.exp(-((lat - a) ** 2 + (lon - b) ** 2) / (2 * s ** 2)) for a, b, h, s in hills) + rng.normal(0, 3, n_st)
    gus = np.array(["가구", "나구", "다구", "라구", "마구"])[(lon * 100).astype(int) % 5]
    st = pd.DataFrame({"station_id": [str(i + 100) for i in range(n_st)], "name": [f"데모대여소{i}" for i in range(n_st)],
                       "gu": gus, "lat": lat, "lon": lon, "docks": rng.integers(5, 25, n_st), "elev": elev})
    dist = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    w = np.exp(-dist / 1500) * np.exp(-np.clip(elev[None, :] - elev[:, None], 0, None) / 15)  # 가깝고 내리막 선호
    w /= w.sum(1, keepdims=True)
    o_counts = rng.multinomial(n_trips, np.exp(-np.clip(elev - 15, 0, None) / 200) / np.exp(-np.clip(elev - 15, 0, None) / 200).sum())
    rows = []
    for i in range(n_st):
        c = rng.multinomial(o_counts[i], w[i]) if o_counts[i] else np.zeros(n_st, int)
        nz = np.nonzero(c)[0]
        rows.append(pd.DataFrame({"o": st.station_id.values[i], "d": st.station_id.values[nz], "n": c[nz]}))
    od = pd.concat(rows, ignore_index=True)
    return st, od, 30, n_trips, 2200.0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="따릉이 언덕 이동격차 분석")
    ap.add_argument("--stations", help="공공자전거 대여소 정보 파일 (.xlsx 또는 .csv)")
    ap.add_argument("--trips", help='공공자전거 대여이력 CSV 경로 패턴, 예: "data/trips/*.csv"')
    ap.add_argument("--elev-csv", help="(선택) 대여소 고도 CSV: station_id, elev")
    ap.add_argument("--dataset", default="srtm30m", help="Open Topo Data 데이터셋 (srtm30m / aster30m)")
    ap.add_argument("--od", help="(선택) 대여이력 대신 쓸 O-D 집계 파일 (저장소의 data/od_2606.csv.gz)")
    ap.add_argument("--save-od", help="(선택) 대여이력을 집계한 O-D 파일을 이 경로에 저장")
    ap.add_argument("--demo", action="store_true", help="가짜 데이터로 작동 테스트")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.demo:
        print("※ DEMO 모드: 가짜 데이터입니다. 결과를 보고서에 쓰지 마세요.")
        st, od, n_days, n_total, avg_m = make_demo()
    else:
        print("[1/4] 대여소 정보 불러오기"); st = load_stations(a.stations)
        print(f"  - 대여소 {len(st):,}개")
        print("[2/4] 고도 붙이기"); st = get_elevations(st, f"{OUT}/elevation_cache.csv", a.dataset, a.elev_csv)
        st = st.dropna(subset=["elev"])
        if a.od:
            print("[3/4] O-D 집계 파일 불러오기"); od, n_days, n_total, avg_m = load_od(a.od)
        else:
            print("[3/4] 대여이력 집계"); od, n_days, n_total, avg_m = load_trips(a.trips)
            if a.save_od:
                save_od(od, n_days, n_total, avg_m, a.save_od); print(f"  - O-D 집계 저장: {a.save_od}")
    print("[4/4] 지형 특성·분석·시각화")
    st = station_terrain_features(st)
    st, sa, ml, pair, tab, imp, top, gu, res = analyze(st, od, n_days, n_total, avg_m)
    print("  - 강건성 검증(부트스트랩·민감도) 중 ...")
    res["robustness"] = robustness(sa.reset_index(drop=True), st_all=st.reset_index(drop=True), n_boot=300 if a.demo else 1000)
    figures(sa, ml, tab, imp, top, gu, res, demo=a.demo)
    st.to_csv(f"{OUT}/station_features.csv", index=False, encoding="utf-8-sig")
    top.to_csv(f"{OUT}/ebike_priority_top30.csv", index=False, encoding="utf-8-sig")
    gu.to_csv(f"{OUT}/district_summary.csv", encoding="utf-8-sig")
    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    write_summary(res, top, gu)
    print(f"\n완료! '{OUT}' 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
