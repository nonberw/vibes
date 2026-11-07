"""Fill docs/overview_numbers.tex (Russian overview) from results/v2/summary.json.
    python scripts/overview_numbers.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "overview_numbers.tex"


def f(x, nd=2):
    try:
        x = float(x[0] if isinstance(x, list) else x)
        return "--" if x != x else f"{x:.{nd}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "--"


def ci(v):
    return f"{f(v)} [{f(v[1])}; {f(v[2])}]" if isinstance(v, list) and len(v) == 3 else f(v)


LOCUS_RU = {"execution-limited": "исполнение", "inference-limited": "вывод", "unreachable belief": "недостижимый belief",
            "mixed": "смешанный", "not identifiable (steering invalid)": "не определяется"}
FAM_RU = {"state": "STATE (истинные координаты)", "rec": "REC (реконструкция пикселей)", "jepa": "JEPA (латенты)"}


def main():
    sp = ROOT / "results" / "v2" / "summary.json"
    if not sp.exists():
        OUT.write_text("\\newcommand{\\OverviewResults}{\\textcolor{gray}{(результаты появятся после завершения сетки)}}\n",
                       encoding="utf-8")
        return
    S = json.loads(sp.read_text())
    rows = []
    for fam in ("state", "rec", "jepa"):
        o = S["configs"].get(f"{fam}_M_gru_train_100000")
        if not o:
            continue
        rows.append(
            f"{FAM_RU[fam]} & {o['n_seeds']} & {f(np.mean(o['geff_slope']))} & {f(np.mean(o['r2_g_trained']))} & "
            f"{ci(o['nat_hi'])} & {ci(o['inv_h_hi'])} & {ci(o['inv_z_hi'])} & "
            f"{LOCUS_RU.get(o['locus_hi'], o['locus_hi'])} / {LOCUS_RU.get(o['locus_lo'], o['locus_lo'])} \\\\")
    table = ("\\begin{table}[h]\\centering\\small\n\\begin{tabular}{lccccccc}\\toprule\n"
             "Семейство & сиды & наклон & $R^2_g$ & $\\rho_{\\rm hi}$ естеств. & $\\rho_{\\rm hi}$ инверсия belief & "
             "$\\rho_{\\rm hi}$ инверсия контекста & локус (верх / низ) \\\\\\midrule\n" + "\n".join(rows) +
             "\n\\bottomrule\\end{tabular}\n\\caption{Главная таблица: 1 = закон, 0 = прилипание к границе; "
             "95\\%-е бутстрап-интервалы в скобках.}\\end{table}\n")
    bullets = []
    for fam in ("state", "rec", "jepa"):
        o = S["configs"].get(f"{fam}_M_gru_train_100000")
        if not o:
            continue
        gate = "прошло" if o["gate_pass"] else "\\textbf{не прошло}"
        bullets.append(
            f"\\item \\textbf{{{FAM_RU[fam]}}}: гейт компетентности {gate} (наклон {f(np.mean(o['geff_slope']))}). "
            f"Естественный отклик вне диапазона сверху $\\rho_{{\\rm hi}}={ci(o['nat_hi'])}$, снизу "
            f"$\\rho_{{\\rm lo}}={ci(o['nat_lo'])}$. Инверсия belief'а: {ci(o['inv_h_hi'])} (верх), "
            f"{ci(o['inv_h_lo'])} (низ); антифизический контроль даёт $g_{{\\rm eff}}={ci(o['inv_h_anti_geff'])}$ "
            f"при цели $-4$. Вердикт по пре-регистрированному правилу: {LOCUS_RU.get(o['locus_hi'], o['locus_hi'])} "
            f"(верх), {LOCUS_RU.get(o['locus_lo'], o['locus_lo'])} (низ).")
    for fam in ("state", "rec", "jepa"):
        if f"H4_{fam}" in S:
            h = S[f"H4_{fam}"]
            bullets.append(f"\\item Масштаб данных, {fam.upper()}: $\\rho_{{\\rm hi}}$ меняется на {ci(h['slope_rho_hi_per_decade'])} "
                           f"за декаду эпизодов, ошибка в диапазоне на {ci(h['slope_test_err_px_per_decade'])} px за декаду.")
    hyp = S.get("pvals_holm", {})
    if hyp:
        sup = sum(1 for v in hyp.values() if v < 0.05)
        bullets.append(f"\\item Подтверждённых пре-регистрированных тестов (Holm, $\\alpha=0{,}05$): {sup} из {len(hyp)}; "
                       f"полная таблица в \\texttt{{results/v2/hypotheses.md}}.")
    body = table + "\\begin{itemize}\n" + "\n".join(bullets) + "\n\\end{itemize}\n"
    OUT.write_text("\\newcommand{\\OverviewResults}{" + body + "}\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
