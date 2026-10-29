# Yield Curve Control and Yen Currency Premia

Python code for my junior independent work at Princeton (advised by Prof. Nobuhiro Kiyotaki, 2025–2026). The paper asks how the Bank of Japan's September 2016 Yield Curve Control (YCC) changed the drivers of the 3-month yen/dollar covered interest parity (CIP) deviation.

The main finding is that the CIP deviation moved from tracking intermediary balance-sheet stress before YCC toward tracking global risk after it. In regime-split regressions, the loading on the US corporate spread (the intermediary stress proxy) flips sign across the 2016 boundary and the VIX loading grows sharply. A pooled interaction Wald test rejects equal coefficients across regimes (F = 6.37, p = 0.002).

## What the script does

`ycc_analysis.py` runs the full analysis end to end.

1. Builds a month-end panel from December 2008 to February 2026 (207 months) and constructs the 3-month CIP deviation and the expected UIP deviation from spot, forward swap points, survey forecasts and 3-month interest rates.
2. Tests for structural breaks with Bai-Perron-type endogenous break selection (`ruptures`, dynamic programming with an L2 cost) and a Chow test at the policy date.
3. Compares means and volatility before and after YCC (Welch, Levene and variance-ratio tests).
4. Estimates regime-split OLS regressions of the CIP deviation on the intermediary stress proxy, S&P financials, the VIX and the lagged spot change, with Newey-West HAC standard errors (4 lags).
5. Runs pooled interaction Wald tests for coefficient stability across the YCC boundary and after the March 2024 exit.
6. Writes regression tables and figures to `output/`.

## Output

| File | Contents |
|---|---|
| `output/stata_style_regressions.txt` | Regime-split regressions and Wald tests, formatted like Stata output |
| `output/fig1_EUIP_Dev_timeseries.png` | Expected UIP deviation over time |
| `output/fig2_CIP_EBP_dual_axis.png` | CIP deviation against the corporate spread |
| `output/fig3_CIP_Dev_shaded.png` | CIP deviation with policy regimes shaded |

## Data

No data is included, since several series come from licensed databases accessed through Princeton. Place the input files in `data/` under the names the script expects.

| File | Series | Source |
|---|---|---|
| `jpy_usd_spot.csv` | JPY per USD spot (DEXJPUS) | FRED |
| `Japan_int.rate_3m.csv` | Japan 3-month interbank rate (IR3TIB01JPM156N) | FRED (OECD) |
| `US_int.rate_3m.csv` | US 3-month T-bill (DTB3) | FRED |
| `US_corporate_bond_spreads.csv` | ICE BofA US corporate OAS (BAMLC0A0CM) | FRED |
| `Vix_data.csv` | VIX (VIXCLS) | FRED |
| `jpy_usd_3m.forward.xlsx` | 3-month JPY/USD forward swap points (JPY3M Curncy) | Bloomberg Terminal |
| `3m-ex-rate-forecast - Sheet1.csv` | 3-month-ahead JPY/USD forecasts | Haver Analytics |
| `S&P 500 EQUAL WEIGHTED Financials (Sector).csv` | S&P 500 equal-weight financials index | S&P Capital IQ |
| `Japan_corporate_bond_spreads.csv` | Japan corporate bond yield (INJPNW) | Finaeon (Global Financial Data) |

## Running it

```
pip install -r requirements.txt
python ycc_analysis.py
```
