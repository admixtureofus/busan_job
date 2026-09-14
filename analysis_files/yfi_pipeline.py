#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
청년 적합 일자리 지수 (2층 구조) — B+C 안
==========================================
1층  지원 가능성 : 셀 공고 중 (4년제 이상 요구 AND 신입 지원 가능) 비율
2층  잔류 유인   : GOMS 로지스틱 회귀 계수를 가중치로 쓴 공고 점수

  STEP1  GOMS 4개년(2016~2019) 부산 표본 통합
  STEP2  부산 잔류(사업체 위치=부산) ~ 일자리 속성 로지스틱 회귀
         -> 오즈비, AUC, 2층 가중치
  STEP3  가중치를 채용공고 25,135건에 적용
  STEP4  직종 x 구·군 셀 집계, K-means
  STEP5  시각화

확정된 규칙
  - 잔류 정의      : 현 직장 사업체 위치 시도 == 부산 (g**1a014 == 2)
  - 임금(공고)     : 연봉환산최소(하한)만 사용, 결측은 채우지 않음
  - 고용형태 결측  : 미기재 1,057건은 분모에서 제외
  - 고용형태 복수  : 정규직이 포함되면 정규직으로 집계
  - 최종 지수      : 1층·2층을 합치지 않고 2축으로 유지

실행
  python yfi_pipeline.py --prep          # GOMS xlsx -> csv 캐시 (최초 1회, 약 1분)
  python yfi_pipeline.py --out ./out
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, roc_curve, silhouette_score,
                             davies_bouldin_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────

BUSAN = 2                      # 소재지_시도 코드에서 부산
MIN_CELL = 100                 # 셀 최소 공고 수
MIN_PAY_N = 30                 # 임금 중위값을 신뢰할 최소 공개 건수

# GOMS 원본. 변수명은 g{YY}1{접미사} 규칙으로 4개년 동일하다.
# 2018년 파일만 컬럼명이 대문자라 별도 처리한다.
GOMS_SRC = [
    (2016, "/mnt/project/GP16_2017200306.xlsx", "xlsx"),
    (2017, "/mnt/project/GP17__2018.XLSX", "xlsx"),
    (2018, "/mnt/project/GP18__2019.XLSX", "xlsx_upper"),
    (2019, "/mnt/user-data/uploads/GP19__2020.SAV", "sav"),
]
GOMS_VARS = ["majorcat", "area", "sex", "graduy", "sq006",
             "a007a_2018", "a010", "a011", "a014", "a021", "a059", "a122", "f006",
             "a297"]   # a297 = 이직 준비 여부 (1 예 / 2 아니오)

# KECO 2018 직업 대분류 — GOMS와 공고 데이터가 같은 체계를 쓴다
KECO_MAJOR = {
    1: "경영·사무·금융·보험", 2: "연구·공학기술", 3: "교육·법률·사회복지·경찰·소방",
    4: "보건·의료", 5: "예술·디자인·방송·스포츠", 6: "미용·여행·숙박·음식·경비·돌봄·청소",
    7: "영업·판매·운전·운송", 8: "건설·채굴", 9: "설치·정비·생산", 10: "농림어업",
}
MAJORCAT = {1: "인문", 2: "사회", 3: "교육", 4: "공학", 5: "자연", 6: "의약", 7: "예체능"}

C = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a",
     "purple": "#7d5bd6", "mute": "#b0ada4", "ink2": "#55544f", "line": "#e6e4de"}
PAL = [C["orange"], C["blue"], C["green"], C["purple"], "#c2410c", "#0f766e"]


def setup_font():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    pick = next((n for n in ["Noto Sans CJK KR", "NanumGothic", "Malgun Gothic",
                             "AppleGothic", "Noto Sans CJK JP"] if n in have), None)
    if pick is None:
        pick = next((n for n in sorted(have)
                     if any(t in n for t in ("CJK", "Gothic", "Nanum"))), None)
    if pick:
        plt.rcParams["font.family"] = pick
    else:
        print("  ! 한글 폰트 없음: apt-get install fonts-nanum")
    plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 130,
                         "savefig.bbox": "tight", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c6c3bb"})


# ────────────────────────────────────────────────────────────
# STEP1  GOMS 통합
# ────────────────────────────────────────────────────────────

def prep_goms(cache="goms"):
    """원본 xlsx/sav 에서 필요한 13개 변수만 뽑아 연도별 csv 로 캐시한다."""
    os.makedirs(cache, exist_ok=True)
    for year, path, kind in GOMS_SRC:
        out = f"{cache}/g{year}.csv"
        if os.path.exists(out):
            print(f"  {year} 캐시 있음"); continue
        if not os.path.exists(path):
            print(f"  ! {year} 원본 없음: {path}"); continue
        p = f"g{str(year)[2:]}1"
        if kind == "sav":
            import pyreadstat
            d, _ = pyreadstat.read_sav(path, usecols=[p + v for v in GOMS_VARS])
            d.columns = [c.replace(p, "") for c in d.columns]
        else:
            up = (kind == "xlsx_upper")
            want = {(p + v).upper() if up else p + v for v in GOMS_VARS}
            d = pd.read_excel(path, engine="calamine",
                              usecols=lambda c, w=want, u=up:
                              (str(c).upper() if u else str(c)) in w)
            d.columns = [(str(c).upper().replace(p.upper(), "").lower() if up
                          else str(c).replace(p, "")) for c in d.columns]
        d["year"] = year
        d.to_csv(out, index=False)
        print(f"  {year} 추출 {d.shape}")


def load_goms(cache="goms"):
    """부산 소재 대학 졸업 + 임금근로자 + 핵심변수 유효 표본만 남긴다."""
    fs = [f"{cache}/g{y}.csv" for y, _, _ in GOMS_SRC if os.path.exists(f"{cache}/g{y}.csv")]
    if not fs:
        raise FileNotFoundError("GOMS 캐시가 없다. --prep 을 먼저 실행할 것")
    g = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)

    # -1 은 모름/무응답 코드
    for c in ["a122", "a011", "a010", "a007a_2018", "f006", "a014"]:
        g[c] = g[c].replace(-1, np.nan)

    n0 = len(g)
    b = g[g["area"] == BUSAN].copy()                      # 부산 소재 대학
    n1 = len(b)
    b = b[b["sq006"] == 1]                                # 임금근로자만
    b = b.dropna(subset=["a014", "a122", "a059", "a007a_2018"])
    b = b[b["a059"].isin([1, 2])]
    b = b[b["a122"] > 0]

    b["stay"] = (b["a014"] == BUSAN).astype(int)          # 종속변수
    b["regular"] = (b["a059"] == 1).astype(int)
    b["log_pay"] = np.log(b["a122"])
    b["firm_size"] = b["a011"].fillna(b["a011"].median())
    b["female"] = (b["sex"] == 2).astype(int)
    b["busan_hs"] = (b["f006"] == BUSAN).astype(int)      # 연고 통제
    b["keco"] = b["a007a_2018"].astype(int)
    b["a297"] = b["a297"]                     # 이직 준비 여부 (잔류 모델용)
    b["major"] = b["majorcat"].astype(int)
    print(f"  전체 {n0:,} -> 부산 소재 대학 {n1:,} -> 분석 표본 {len(b):,}")
    print(f"  연도별 {b.groupby('year').size().to_dict()}")
    print(f"  부산 잔류율(사업체 위치) {b['stay'].mean():.3f}")
    return b


# ────────────────────────────────────────────────────────────
# STEP2  로지스틱 회귀 -> 2층 가중치
# ────────────────────────────────────────────────────────────

INTEREST = ["regular", "log_pay"]                 # 지수에 들어갈 변수
CONTROL = ["firm_size", "female", "busan_hs"]     # 편향 제거용, 지수에 넣지 않음


def _design(b, extra_year=True):
    X = pd.concat([
        b[INTEREST + CONTROL],
        pd.get_dummies(b["keco"].map(KECO_MAJOR), prefix="직종", drop_first=True),
        pd.get_dummies(b["major"].map(MAJORCAT), prefix="전공", drop_first=True),
    ], axis=1)
    if extra_year:
        X = pd.concat([X, pd.get_dummies(b["year"], prefix="졸업", drop_first=True)], axis=1)
    return sm.add_constant(X.astype(float))


def _table(res):
    ci = res.conf_int()
    return pd.DataFrame({"계수": res.params, "표준오차": res.bse,
                         "오즈비": np.exp(res.params), "p값": res.pvalues,
                         "CI하한": np.exp(ci[0]), "CI상한": np.exp(ci[1])}).round(4)


def fit_location(b: pd.DataFrame):
    """
    [발표 본론용] 부산 소재 대학 졸업 임금근로자 전원.
    종속변수 = 현 직장 사업체 위치가 부산인가.

    주의: 이 회귀의 계수는 지수 가중치로 쓰지 않는다.
    종속변수가 '일자리의 위치'이고 설명변수가 '그 일자리의 질'이라,
    모델이 재는 것은 "좋은 일자리가 어디에 있는가"이다.
    실제로 정규직·임금·기업규모의 오즈비가 모두 1 미만으로 나온다.
    이것은 오류가 아니라 결과이며, 발표의 핵심 근거로 쓴다.
    """
    X = _design(b)
    res = sm.Logit(b["stay"], X).fit(disp=0, maxiter=200)
    pred = res.predict(X)
    auc = roc_auc_score(b["stay"], pred)
    print(f"  [위치 모델] n={len(b):,}  Pseudo R²={res.prsquared:.4f}  AUC={auc:.4f}")
    tab = _table(res)
    print(tab.loc[INTEREST + CONTROL].to_string())
    return res, tab, auc, pred


def fit_retention(b: pd.DataFrame):
    """
    [가중치 산출용] 부산에서 일하는 사람만.
    종속변수 = 이직 준비를 하고 있지 않은가 (a297 == 2).

    표본을 부산 근무자로 고정하면 지역 효과가 상수가 되므로,
    "부산 안에서 어떤 자리가 사람을 붙잡는가"를 분리해서 물을 수 있다.
    """
    w = b[(b["a014"] == BUSAN) & b["a297"].isin([1, 2])].copy()
    w["keep"] = (w["a297"] == 2).astype(int)
    X = _design(w)
    res = sm.Logit(w["keep"], X).fit(disp=0, maxiter=300)
    pred = res.predict(X)
    auc = roc_auc_score(w["keep"], pred)

    sd = w[INTEREST].std()
    std_coef = res.params[INTEREST] * sd
    weights = (std_coef / std_coef.abs().sum()).to_dict()

    print(f"  [잔류 모델] n={len(w):,}  이직준비 안 함 {w['keep'].mean():.3f}  "
          f"Pseudo R²={res.prsquared:.4f}  AUC={auc:.4f}")
    tab = _table(res)
    print(tab.loc[INTEREST + CONTROL].to_string())
    print(f"  회귀 가중치 {({k: round(v, 3) for k, v in weights.items()})}")
    if min(weights.values()) < 0:
        print("  ! 가중치에 음수가 있다. 지수 방향을 확인할 것")
    return res, tab, auc, pred, weights, w


def entropy_weights(M: pd.DataFrame) -> dict:
    """
    [민감도용] 엔트로피 가중법. 종속변수 없이, 지표가 대안들 사이에서
    얼마나 다양하게 퍼져 있는지로 가중치를 정한다.
    퍼짐이 큰(= 구분에 기여하는) 지표가 큰 가중치를 받는다.
    """
    P = (M - M.min()) / (M.max() - M.min() + 1e-12)   # 0~1 정규화
    P = P + 1e-6
    P = P / P.sum()
    n = len(P)
    e = -(P * np.log(P)).sum() / np.log(n)             # 엔트로피
    d = 1 - e                                          # 다양성
    w = (d / d.sum()).to_dict()
    print(f"  엔트로피 가중치 {({k: round(v, 3) for k, v in w.items()})}")
    return w


# ────────────────────────────────────────────────────────────
# STEP3  공고 점수화
# ────────────────────────────────────────────────────────────

def load_postings(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.columns = [c.strip().lstrip("\ufeff") for c in d.columns]

    # 고용형태: 미기재는 결측 처리(분모에서 제외).
    # 복수 고용형태(3,220건) 중 정규직이 포함된 3,055건은 이미 '정규직'으로
    # 대표값이 잡혀 있어, 확정 규칙과 동일한 결과가 된다.
    d["고용형태"] = d["고용형태"].replace("미기재", np.nan)
    d["regular"] = np.where(d["고용형태"].isna(), np.nan,
                            (d["고용형태"] == "정규직").astype(float))
    d["복수확인필요"] = ((d["고용형태복수"] == "Y") &
                        (d["고용형태"] != "정규직")).astype(int)

    d["edu_univ"] = d["요구학력"].isin(["대학교졸", "석사이상"]).astype(int)
    d["newbie_ok"] = d["경력요건"].isin(["신입", "무관"]).astype(int)
    d["지원가능"] = d["edu_univ"] * d["newbie_ok"]          # 1층

    # 임금: 하한만, 결측은 채우지 않는다
    pay = d["연봉환산최소"].copy()
    lo, hi = pay.quantile([0.01, 0.99])
    d["pay"] = pay.clip(lo, hi)
    d["pay_open"] = d["pay"].notna().astype(int)
    d["log_pay"] = np.log(d["pay"] / 12)                    # GOMS는 월 만원 단위
    return d


def score_layer2(d: pd.DataFrame, w_main: dict, w_alt: dict):
    """2층 점수. 임금이 공개되고 고용형태가 유효한 공고에만 부여한다."""
    ok = d["pay_open"].eq(1) & d["regular"].notna()
    z = pd.DataFrame({
        "regular": (d.loc[ok, "regular"] - d.loc[ok, "regular"].mean()) / d.loc[ok, "regular"].std(),
        "log_pay": (d.loc[ok, "log_pay"] - d.loc[ok, "log_pay"].mean()) / d.loc[ok, "log_pay"].std(),
    })

    def mk(w):
        r = sum(z[k] * w[k] for k in w)
        return (r - r.min()) / (r.max() - r.min()) * 100

    d["잔류유인"] = np.nan; d["잔류유인_엔트로피"] = np.nan; d["잔류유인_동일"] = np.nan
    d.loc[ok, "잔류유인"] = mk(w_main)
    d.loc[ok, "잔류유인_엔트로피"] = mk(w_alt)
    d.loc[ok, "잔류유인_동일"] = mk({k: 0.5 for k in w_main})

    r1 = spearmanr(d.loc[ok, "잔류유인"], d.loc[ok, "잔류유인_엔트로피"]).correlation
    r2 = spearmanr(d.loc[ok, "잔류유인"], d.loc[ok, "잔류유인_동일"]).correlation
    print(f"  2층 점수 {int(ok.sum()):,}건 부여")
    print(f"  순위상관 — 회귀 vs 엔트로피 {r1:.3f} / 회귀 vs 동일가중 {r2:.3f}")
    return d, {"vs_entropy": float(r1), "vs_equal": float(r2)}


# ────────────────────────────────────────────────────────────
# STEP4  셀 집계 + K-means
# ────────────────────────────────────────────────────────────

CLUSTER_FEATS = ["지원가능률", "정규직비율", "임금중위"]


def build_cells(d: pd.DataFrame, score: bool = True) -> pd.DataFrame:
    s = d[(d["시군구"] != "미상") & (d["직종대분류"] != "미분류")].copy()
    g = s.groupby(["시군구", "직종대분류"]).agg(
        공고수=("공고ID", "count"),
        지원가능률=("지원가능", lambda x: x.mean() * 100),
        정규직비율=("regular", lambda x: x.mean() * 100),      # 결측 자동 제외
        임금중위=("pay", "median"),
        공개n=("pay_open", "sum"),
        **({"잔류유인": ("잔류유인", "mean"),
            "잔류유인_엔트로피": ("잔류유인_엔트로피", "mean")} if score else {}),
    ).reset_index()
    g = g[g["공고수"] >= MIN_CELL].copy()
    g["임금신뢰"] = (g["공개n"] >= MIN_PAY_N).astype(int)
    print(f"  셀 {len(g)}개 (공고 {MIN_CELL}건 이상) / "
          f"임금 표본 {MIN_PAY_N}건 미만 {int((1 - g['임금신뢰']).sum())}개")
    return g


def choose_k(X, kmax=6, min_size=3, seed=42):
    rows = []
    for k in range(2, kmax + 1):
        lab = KMeans(k, n_init=200, random_state=seed).fit_predict(X)
        rows.append({"k": k, "실루엣": round(silhouette_score(X, lab), 3),
                     "DB지수": round(davies_bouldin_score(X, lab), 3),
                     "최소군집": int(np.bincount(lab).min())})
    return pd.DataFrame(rows)


def stability_ari(X, k, n_boot=100, seed=42):
    rng = np.random.default_rng(seed)
    base = KMeans(k, n_init=100, random_state=seed).fit_predict(X)
    out = []
    for _ in range(n_boot):
        i = rng.choice(len(X), len(X), replace=True)
        km = KMeans(k, n_init=20, random_state=int(rng.integers(1e6))).fit(X[i])
        out.append(adjusted_rand_score(base, km.predict(X)))
    return float(np.mean(out))


def cluster(cells: pd.DataFrame, k=None, seed=42, strict_pay=False):
    c = cells[cells["임금신뢰"] == 1].copy() if strict_pay else cells.copy()
    c["임금중위"] = c["임금중위"].fillna(c["임금중위"].median())
    X = StandardScaler().fit_transform(c[CLUSTER_FEATS])
    diag = choose_k(X)
    if k is None:
        ok = diag[diag["최소군집"] >= 3]
        ok = ok if len(ok) else diag
        k = int(ok.sort_values("실루엣", ascending=False).iloc[0]["k"])
    lab = KMeans(k, n_init=300, random_state=seed).fit_predict(X)
    c["군집"] = lab
    sil = silhouette_score(X, lab)
    ari = stability_ari(X, k, seed=seed)
    prof = c.groupby("군집")[CLUSTER_FEATS].mean().round(1)
    prof["셀수"] = c.groupby("군집").size()
    prof["공고수"] = c.groupby("군집")["공고수"].sum()
    print(f"  k={k}  실루엣={sil:.3f}  부트스트랩 ARI={ari:.3f}")
    print(diag.to_string(index=False))
    print(prof.to_string())
    return c, prof, diag, k, sil, ari, X


# ────────────────────────────────────────────────────────────
# STEP5  시각화
# ────────────────────────────────────────────────────────────

def fig_odds(tab, out, fname, title, xlab):
    keys = INTEREST + CONTROL
    t = tab.loc[keys].sort_values("오즈비")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    y = range(len(t))
    col = [C["orange"] if i in INTEREST else C["mute"] for i in t.index]
    ax.errorbar(t["오즈비"], y,
                xerr=[t["오즈비"] - t["CI하한"], t["CI상한"] - t["오즈비"]],
                fmt="o", color="none", ecolor="#c9c6be", elinewidth=1.6, capsize=3)
    ax.scatter(t["오즈비"], y, s=80, color=col, zorder=3, edgecolor="white", lw=1.2)
    ax.axvline(1, color=C["line"], lw=1.6)
    names = {"regular": "정규직", "log_pay": "임금(로그)", "firm_size": "기업 규모",
             "female": "여성", "busan_hs": "부산 고교 출신"}
    ax.set_yticks(list(y)); ax.set_yticklabels([names.get(i, i) for i in t.index])
    for i, (v, p) in enumerate(zip(t["오즈비"], t["p값"])):
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        ax.text(v, i + 0.28, f"{v:.2f}{star}", ha="center", fontsize=8.5, color=C["ink2"])
    ax.set_xlabel(xlab)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    fig.savefig(f"{out}/{fname}"); plt.close(fig)


def fig_roc(y, pred, auc, out, fname, label):
    fpr, tpr, _ = roc_curve(y, pred)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot(fpr, tpr, color=C["orange"], lw=2.2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=C["mute"], lw=1.2)
    ax.set_xlabel("거짓 양성률"); ax.set_ylabel("참 양성률")
    ax.set_title(f"{label} 성능", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(f"{out}/{fname}"); plt.close(fig)


def fig_two_axis(cells, prof, out):
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for cid, s in cells.groupby("군집"):
        ax.scatter(s["지원가능률"], s["잔류유인"], s=s["공고수"] / 6 + 45,
                   color=PAL[cid % len(PAL)], alpha=0.85, edgecolor="white", lw=1.3,
                   label=f"군집 {cid} ({len(s)}셀)")
    for _, r in cells.nlargest(10, "공고수").iterrows():
        ax.annotate(f"{r['시군구']}·{str(r['직종대분류'])[:5]}",
                    (r["지원가능률"], r["잔류유인"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7.5, color=C["ink2"])
    ax.axvline(cells["지원가능률"].median(), color=C["line"], lw=1.3)
    ax.axhline(cells["잔류유인"].median(), color=C["line"], lw=1.3)
    ax.margins(0.16)
    q = [(0.015, .985, "left", "top", "지원은 어렵지만\n남을 이유는 있다"),
         (.985, .985, "right", "top", "지원도 되고\n남을 이유도 있다"),
         (0.015, 0.015, "left", "bottom", "지원도 어렵고\n남을 이유도 없다"),
         (.985, 0.015, "right", "bottom", "지원은 되지만\n남을 이유가 없다")]
    for fx, fy, ha, va, t in q:
        ax.text(fx, fy, t, transform=ax.transAxes, ha=ha, va=va,
                fontsize=8.5, color=C["mute"], linespacing=1.4)
    ax.set_xlabel("1층 · 지원 가능률 (%)"); ax.set_ylabel("2층 · 잔류 유인 점수")
    ax.set_title("지원할 수 있는 자리와 남을 이유가 있는 자리는 다르다",
                 loc="left", fontsize=11, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.savefig(f"{out}/03_two_axis.png"); plt.close(fig)


def fig_k(diag, k, out):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), gridspec_kw={"wspace": .3})
    for ax, col, ttl, best in [(axes[0], "실루엣", "실루엣 (높을수록 좋음)", "max"),
                               (axes[1], "DB지수", "Davies-Bouldin (낮을수록 좋음)", "min")]:
        ax.plot(diag["k"], diag[col], "o-", color=C["blue"], lw=2, ms=6)
        i = diag[col].idxmax() if best == "max" else diag[col].idxmin()
        ax.plot(diag.loc[i, "k"], diag.loc[i, col], "o", color=C["orange"],
                ms=11, mfc="none", mew=2.2)
        ax.axhline(0.5, color=C["line"], lw=1.2, ls="--") if col == "실루엣" else None
        ax.set_xticks(diag["k"]); ax.set_xlabel("k")
        ax.set_title(ttl, fontsize=10, fontweight="bold")
    fig.suptitle(f"선택된 k = {k}", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(f"{out}/04_k.png"); plt.close(fig)


def fig_profile(prof, out, ref=None):
    """ref = 셀 원표. 정규화 기준을 전체 셀로 잡아야 군집이 2개일 때도
    막대가 0/1로 무너지지 않는다."""
    feats = CLUSTER_FEATS
    B = ref[feats] if ref is not None else prof[feats]
    norm = ((prof[feats] - B.min()) / (B.max() - B.min() + 1e-9)).clip(0, 1)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    w, n = 0.8 / len(feats), len(prof)
    for j, f in enumerate(feats):
        ax.bar(np.arange(n) + j * w, norm[f], width=w, label=f,
               color=PAL[j % len(PAL)], alpha=0.88)
        for i, v in enumerate(prof[f]):
            ax.text(i + j * w, norm[f].iloc[i] + .02, f"{v:.0f}",
                    ha="center", fontsize=8, color=C["ink2"])
    ax.set_xticks(np.arange(n) + w)
    ax.set_xticklabels([f"군집 {i}\n({int(prof.loc[i,'셀수'])}셀 · "
                        f"{int(prof.loc[i,'공고수']):,}건)" for i in prof.index])
    ax.set_yticks([]); ax.set_ylim(0, 1.18)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.set_title("군집 프로파일 (막대 높이는 전체 셀 대비 상대 위치, 숫자는 실제값)",
                 loc="left", fontsize=11, fontweight="bold", pad=10)
    fig.savefig(f"{out}/05_profile.png"); plt.close(fig)


# ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/user-data/uploads/부산_공고_분석용.csv")
    ap.add_argument("--cache", default="goms")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--prep", action="store_true", help="GOMS 원본 추출만 수행")
    ap.add_argument("--strict-pay", action="store_true",
                    help="임금 공개 30건 미만 셀 제외 (민감도 분석)")
    a = ap.parse_args()

    if a.prep:
        prep_goms(a.cache); return
    os.makedirs(a.out, exist_ok=True)
    setup_font()

    print("[STEP1] GOMS 통합")
    if not os.path.exists(f"{a.cache}/g2019.csv"):
        prep_goms(a.cache)
    b = load_goms(a.cache)

    print("\n[STEP2-a] 위치 모델 (발표 본론용)")
    res_loc, tab_loc, auc_loc, pred_loc = fit_location(b)

    print("\n[STEP2-b] 잔류 모델 (가중치 산출용)")
    res_ret, tab_ret, auc_ret, pred_ret, w_main, keep_df = fit_retention(b)

    print("\n[STEP3] 공고 점수화")
    d = load_postings(a.csv)
    cells_raw = build_cells(d, score=False)
    w_alt = entropy_weights(cells_raw[["정규직비율", "임금중위"]].dropna()
                            .rename(columns={"정규직비율": "regular", "임금중위": "log_pay"}))
    d, rhos = score_layer2(d, w_main, w_alt)

    print("\n[STEP4] 셀 집계 및 K-means")
    cells = build_cells(d)
    cells, prof, diag, k, sil, ari, X = cluster(cells, k=a.k, strict_pay=a.strict_pay)

    print("\n[STEP5] 시각화")
    fig_odds(tab_loc, a.out, "01_odds_location.png",
             "무엇이 부산에 남는 것과 함께 움직이는가 (위치 모델)",
             "오즈비 (1보다 크면 일자리가 부산에 있을 확률이 높다)")
    fig_odds(tab_ret, a.out, "01b_odds_retention.png",
             "부산 안에서 무엇이 사람을 붙잡는가 (잔류 모델)",
             "오즈비 (1보다 크면 이직 준비를 하지 않을 확률이 높다)")
    fig_roc(b["stay"], pred_loc, auc_loc, a.out, "02_roc_location.png", "위치 모델")
    fig_roc(keep_df["keep"], pred_ret, auc_ret, a.out, "02b_roc_retention.png", "잔류 모델")
    fig_two_axis(cells, prof, a.out)
    fig_k(diag, k, a.out)
    fig_profile(prof, a.out, ref=cells)

    tab_loc.to_csv(f"{a.out}/logit_location.csv", encoding="utf-8-sig")
    tab_ret.to_csv(f"{a.out}/logit_retention.csv", encoding="utf-8-sig")
    cells.to_csv(f"{a.out}/cells_clustered.csv", index=False, encoding="utf-8-sig")
    prof.to_csv(f"{a.out}/cluster_profile.csv", encoding="utf-8-sig")
    d[["공고ID", "시군구", "직종대분류", "지원가능", "regular", "pay",
       "잔류유인", "잔류유인_엔트로피", "잔류유인_동일"]].to_csv(f"{a.out}/posting_scores.csv",
                                  index=False, encoding="utf-8-sig")
    json.dump({"weights_regression": w_main, "weights_entropy": w_alt,
               "auc_location": auc_loc, "auc_retention": auc_ret,
               "n_goms": int(len(b)), "n_retention": int(len(keep_df)),
               "stay_rate": float(b["stay"].mean()),
               "silhouette": float(sil), "ari": float(ari),
               "rank_corr": rhos, "k": int(k)},
              open(f"{a.out}/summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n완료 -> {a.out}")


if __name__ == "__main__":
    main()
