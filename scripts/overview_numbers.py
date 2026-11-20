"""Fill docs/overview_numbers.tex (Russian overview) from results/v2/summary.json.
    python scripts/overview_numbers.py"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "overview_numbers.tex"
FAMS = ("state", "rec", "jepa")


def f(x, nd=2):
    try:
        x = float(x[0] if isinstance(x, list) else x)
        return "--" if x != x else f"{x:.{nd}f}".replace(".", "{,}")
    except (TypeError, ValueError):
        return "--"


def ci(v):
    return f"{f(v)} [{f(v[1])}; {f(v[2])}]" if isinstance(v, list) and len(v) == 3 else f(v)


def main():
    sp = ROOT / "results" / "v2" / "summary.json"
    if not sp.exists():
        OUT.write_text("\\newcommand{\\OverviewResults}{\\textcolor{gray}{(результаты появятся после завершения сетки)}}\n",
                       encoding="utf-8")
        return
    S = json.loads(sp.read_text())
    C = S["configs"]
    o = {k: C[f"{k}_M_gru_train_100000"] for k in FAMS}
    ce = {k: S[f"ceiling_{k}"] for k in FAMS}
    name = {"state": "STATE", "rec": "REC", "jepa": "JEPA"}
    rows = []
    for k in FAMS:
        h = ce[k]
        rows.append(
            f"{name[k]} & {f(np.mean(o[k]['geff_slope']))} & {f(np.mean(o[k]['test_err_px']), 1)} & "
            f"{f(np.mean(o[k]['r2_g_trained']))} & {ci(o[k]['inf_hi'])} & {ci(o[k]['lambda_nat_hi'])} & "
            f"{ci(o[k]['lambda_nat_lo'])} & {f(h['lambda_hold_lo']['nat'])} / {f(h['lambda_hold_lo']['inv_h'])} "
            f"/ {f(h['lambda_hold_lo']['wide_ceiling'])} \\\\")
    table = ("\\begin{table}[h]\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{lccccccc}\\toprule\n"
             "Семейство & гейт & ошибка, px & $R^2_g$ & чтение belief, верх & $\\lambda$ верх & $\\lambda$ низ & "
             "$\\lambda$ низ: ест./инв./WIDE \\\\\\midrule\n" + "\n".join(rows) +
             "\n\\bottomrule\\end{tabular}\n\\caption{Основная конфигурация (3 сида). Гейт~--- наклон $g_{\\rm eff}$ по $g$ "
             "внутри диапазона (нужно $\\ge 0{,}5$). Чтение belief~--- наклон линейной пробы вне диапазона "
             "(1~--- отслеживает, 0~--- прилипает). $\\lambda$~--- доля закона в предсказании (1~--- закон, "
             "0~--- прилипание к границе), метрика добавлена после пре-регистрации. В скобках 95\\%-е "
             "бутстрап-интервалы.}\\end{table}\n")
    hyp = S.get("pvals_holm", {})
    sup = sum(1 for v in hyp.values() if v < 0.05)
    st = o["state"]
    d = lambda k, n, key: f(np.mean(C[f"{k}_M_gru_train_{n}"][key]) if len(C[f"{k}_M_gru_train_{n}"][key]) != 3
                            else C[f"{k}_M_gru_train_{n}"][key], 2 if "lambda" in key else 1)
    pc = lambda v: f"{100 * v[0]:.0f}\\,\\% [{100 * v[1]:.0f}; {100 * v[2]:.0f}]"
    b = [
        f"\\item \\textbf{{Пре-регистрация в основном не подтвердилась.}} Из {len(hyp)} подтверждающих тестов поправку "
        f"Холма проходят {sup}: декодируемость и валидность записи для STATE, проверка восприятия для REC и JEPA. "
        "Пре-регистрированная оценка (кривизна предсказанной высоты) оказалась слишком шумной выше диапазона и "
        "смещённой ниже него. Она не различает <<прилипание>> и <<закон>>, а правило локуса выдаёт метки без "
        "устойчивой картины. Это промах дизайна, он честно описан в ограничениях статьи.",
        "\\item \\textbf{Все три семейства компетентны} внутри диапазона (гейт пройден), гравитация читается из "
        f"belief'а: $R^2={f(np.mean(st['r2_g_trained']))}$ для STATE против {f(np.mean(st['r2_g_untrained']))} у "
        "необученной динамики.",
        f"\\item \\textbf{{Восприятие не виновато}}: на кадрах вне диапазона позиции читаются так же точно "
        f"(добавка {ci(o['rec']['perception_ood_excess_px'])} px у REC, {ci(o['jepa']['perception_ood_excess_px'])} px у JEPA).",
        f"\\item \\textbf{{Belief отслеживает гравитацию вне диапазона, но занижает её примерно вдвое}}: наклон пробы "
        f"{ci(st['inf_hi'])} сверху и {ci(st['inf_lo'])} снизу (STATE), у пиксельных семейств так же. Это самое точное "
        "число в работе. Оговорка: проба на замороженных признаках V-JEPA~2 занижает в похожей степени, так что часть "
        "эффекта может быть геометрией кода, а не <<неверием>> модели.",
        f"\\item \\textbf{{Модели не прилипают и не следуют закону}} (разведочная метрика $\\lambda$). Выше диапазона "
        f"предсказания проходят около половины пути от прилипания к закону ({f(st['lambda_nat_hi'])} / "
        f"{f(o['rec']['lambda_nat_hi'])} / {f(o['jepa']['lambda_nat_hi'])} для STATE / REC / JEPA), ниже диапазона "
        f"остаются у границы ({f(st['lambda_nat_lo'])} / {f(o['rec']['lambda_nat_lo'])} / {f(o['jepa']['lambda_nat_lo'])}).",
        f"\\item \\textbf{{Провал поделён между выводом и исполнением}} (разведочно). У STATE, единственного семейства с "
        f"валидной записью, запись истинной гравитации в belief поднимает $\\lambda$ выше диапазона с "
        f"{f(st['lambda_nat_hi'])} до {f(st['lambda_steer_hi'])}, это уровень модели-близнеца WIDE, обученной на "
        f"$g\\in[1,12]$ ({f(ce['state']['lambda_nat_hi'][1])}). Ниже диапазона даже оптимизированный belief закрывает "
        f"лишь {pc(ce['state']['gap_closed_inv_h_lo'])} разрыва до WIDE (REC {pc(ce['rec']['gap_closed_inv_h_lo'])}, "
        f"JEPA {pc(ce['jepa']['gap_closed_inv_h_lo'])}).",
        f"\\item \\textbf{{Гипотеза <<масштаб не покупает экстраполяцию>> не подтвердилась.}} Ошибка внутри диапазона "
        f"падает примерно на 2 px за декаду данных (STATE: {d('state', '1000', 'test_err_px')} / "
        f"{d('state', '8000', 'test_err_px')} / {f(np.mean(st['test_err_px']), 1)} px при 1k / 8k / 100k эпизодов), а "
        f"$\\lambda$ выше диапазона при этом растёт ({d('state', '1000', 'lambda_nat_hi')} / "
        f"{d('state', '8000', 'lambda_nat_hi')} / {f(st['lambda_nat_hi'])}). Размер сети динамики не помог, линейная "
        "рекуррентность (LRU) ухудшила.",
        f"\\item \\textbf{{Энергию не сохраняет никто}}: медианная относительная ошибка энергии растёт с "
        f"{f(st['drift_E_median_by_chunk'][0])} до {f(st['drift_E_median_by_chunk'][-1])} за 80 кадров (STATE). "
        "Линейной ручки для горизонтальной скорости не нашлось.",
    ]
    body = table + "\\begin{itemize}\n" + "\n".join(b) + "\n\\end{itemize}\n"
    OUT.write_text("\\newcommand{\\OverviewResults}{" + body + "}\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
