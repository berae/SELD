"""Build descriptive result tables from saved evidence; never rescore or train.

Run from any directory. Uses Python standard library only.
"""
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'reports/aligned_20260904'
OUT = ROOT / 'reports/catalog_20260904'
METRICS = ('ER20', 'F20', 'LE_CD', 'LR_CD', 'SELD_LR')
KEY = ('family', 'context', 'variant', 'weight', 'seed', 'split')
GROUP = ('family', 'context', 'variant', 'weight')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def key(row, fields=KEY):
    return tuple(row[f] for f in fields)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, content):
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(content)


def summarize(rows, metric, scale=1.0, digits=3):
    values = [float(r[metric]) * scale for r in rows]
    return f'{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def recipe(row):
    family = 'einv2' if row['family'] == 'EINV2' else 'multi_accdoa'
    variant = row['variant'].split('_')[0].replace('C0.1', 'C0')
    extension = 'yaml' if family == 'einv2' else 'json'
    return f'configs/{family}/{variant}.{extension}'


def label(group):
    family, context, variant, weight = group
    return f'{family} {variant}'


def main():
    all_runs = read_csv(BASE / 'aligned_run_metrics.csv')
    runs = [r for r in all_runs if r['profile'] == 'dcase2023_micro']
    evidence = {key(r): r for r in read_csv(BASE / 'evidence_index.csv')}
    assert len(runs) == len({key(r) for r in runs}) == len(evidence) == 84
    assert {key(r) for r in runs} == set(evidence)
    audit = json.loads((OUT / 'source_audit.json').read_text(encoding='utf-8'))
    audit_rows = {key(r): r for r in audit['rows']}
    assert len(audit_rows) == 84
    # Check local source-table bytes against the freshly read server hashes.
    checked = []
    for source in audit['source_tables']:
        name = Path(source['path']).name
        path = (BASE if name.startswith('aligned_') else OUT) / name
        if not path.exists():
            continue
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], name
        checked.append(name)
    groups = defaultdict(list)
    for row in runs:
        groups[key(row, GROUP) + (row['split'],)].append(row)
    aggregate = {(key(r, GROUP) + (r['split'],)): r for r in read_csv(BASE / 'aligned_aggregate_metrics.csv') if r['profile'] == 'dcase2023_micro'}
    for group, rows in groups.items():
        assert sorted(int(r['seed']) for r in rows) == [2026, 2027, 2028]
        for metric in METRICS:
            values = [float(r[metric]) for r in rows]
            assert abs(statistics.mean(values) - float(aggregate[group][metric + '_mean'])) < 1e-10
            assert abs(statistics.stdev(values) - float(aggregate[group][metric + '_sample_sd'])) < 1e-10
        for row in rows:
            score = (float(row['ER20']) + 1 - float(row['F20']) + float(row['LE_CD']) / 180 + 1 - float(row['LR_CD'])) / 4
            assert abs(score - float(row['SELD_LR'])) < 1e-10
    ordered = sorted({g[:-1] for g in groups})
    main_table, delta_table, index = [], [], []
    for group in ordered:
        ev, va = groups[group + ('evaluation',)], groups[group + ('validation',)]
        main_table.append([label(group), group[1], group[3], '3/3', summarize(ev, 'LE_CD'), summarize(ev, 'LR_CD', 100, 2),
                           summarize(ev, 'F20', 100, 2), summarize(ev, 'ER20', digits=4), summarize(ev, 'SELD_LR', digits=4), summarize(va, 'SELD_LR', digits=4)])
        base_variant = 'C0.1' if group[:2] == ('EINV2', 'causal') else 'E0' if group[0] == 'EINV2' else 'C0' if group[1] == 'causal' else 'A0'
        baseline = {r['seed']: r for r in groups[(group[0], group[1], base_variant, '0', 'evaluation')]}
        if group[2] != base_variant:
            deltas = [{m: float(r[m]) - float(baseline[r['seed']][m]) for m in METRICS} for r in ev]
            delta_table.append([label(group), base_variant, summarize(deltas, 'LE_CD'), summarize(deltas, 'LR_CD', 100, 2),
                                summarize(deltas, 'F20', 100, 2), summarize(deltas, 'SELD_LR', digits=4),
                                str(sum(d['LE_CD'] < 0 for d in deltas)) + '/3', str(sum(d['SELD_LR'] < 0 for d in deltas)) + '/3'])
        for row in sorted(ev + va, key=lambda r: (r['seed'], r['split'])):
            source, check = evidence[key(row)], audit_rows[key(row)]
            assert all(check[k]['exists'] for k in ('config', 'checkpoint', 'source'))
            assert not check['predictions']['missing'] and not check['predictions']['extra']
            assert source['checkpoint'] == row['checkpoint'] and source['prediction_dir'] == row['prediction_dir']
            index.append({**{f: row[f] for f in KEY}, 'dataset': 'TAU2020 FOA', 'train_folds': '2,3,4,5,6',
                          'validation_fold': 1, 'status': 'SCORED_SAVED_PREDICTIONS', 'profile': row['profile'],
                          'run_id': source['run_id'], 'portable_recipe_not_historical_config': recipe(row),
                          'historical_config': row['config'], 'checkpoint': row['checkpoint'],
                          'prediction_dir': row['prediction_dir'], 'original_result': source['source'],
                          'checkpoint_bytes': check['checkpoint']['bytes'], 'prediction_files': check['predictions']['files'],
                          **{m: row[m] for m in METRICS}})
    write_csv(OUT / 'run_index.csv', index)
    run_display = []
    for row in [r for r in index if r['split'] == 'evaluation']:
        val = next(r for r in index if all(r[f] == row[f] for f in GROUP + ('seed',)) and r['split'] == 'validation')
        run_display.append([row['family'], row['variant'], row['seed'], row['run_id'],
                            f"{float(row['LE_CD']):.3f}", f"{100 * float(row['LR_CD']):.2f}",
                            f"{float(row['SELD_LR']):.4f}", f"{float(val['SELD_LR']):.4f}"])
    write_text(OUT / 'RUN_INDEX.md', '# 逐 seed 实验列表\n\n42 个 run、84 个 run/split；均为 TAU2020，训练 folds2–6、validation fold1、evaluation 200 条。\n\n'
        '状态为已有 checkpoint / 导出预测 / 评分，并非本次重训。完整 config、checkpoint、prediction、result 绝对路径见 [run_index.csv](run_index.csv)。\n\n'
        + table(['Family', 'Variant', 'Seed', 'Run ID', 'Eval LE_CD °', 'Eval LR_CD %', 'Eval SELD', 'Val SELD'], run_display) + '\n')
    motion = read_csv(OUT / 'completion_motion_per_run.csv')
    assert len(motion) == 168
    motion_table = []
    for group in ordered:
        for stratum in ('static', 'dynamic'):
            rows = [r for r in motion if (r['family'], r['context'], r['variant']) == group[:3] and r['split'] == 'evaluation' and r['motion_group'] == stratum]
            assert len(rows) == 3
            # Diagnostics must refer to the exact predictions in the global table.
            for row in rows:
                matching = next(r for r in runs if r['family'] == row['family'] and r['variant'] == row['variant'] and r['seed'] == row['seed'] and r['split'] == row['split'])
                assert matching['prediction_dir'] == row['prediction_dir']
            motion_table.append([label(group), stratum, int(float(rows[0]['references'])),
                                 summarize(rows, 'mean_localization_error_deg_micro'), summarize(rows, 'matched_recall_micro', 100, 2),
                                 summarize(rows, 'recall_at_threshold_micro', 100, 2)])
    title = '# 实验结果总表\n\n'
    scope = '''## Material Passport

- Origin Skill / Mode: experiment-agent / validate
- Origin Date: 2026-09-04
- Verification Status: ANALYZED
- Version Label: experiment_catalog_v1

## 口径与范围

当前主矩阵：**14 个变体组 × 3 seeds = 42 个 run；validation / evaluation 共 84 组预测**。本表展开已有结果，不新增训练或推理。全部是 TAU2020 FOA，train folds2–6、validation fold1（100 条）、evaluation（200 条）；seeds 2026/2027/2028。

统一主口径为 `dcase2023_micro`、完整 60 秒、1 秒 block、20° detection threshold；不是把数据集换成 DCASE2023。均值 ± sample SD（ddof=1），SD **不是**置信区间；未做显著性检验。ER / LE / SELD 越低越好，F20 / LR_CD 越高越好。F20 带空间条件，不等于纯 SED F1；纯 SED F1 和 mAP 尚无可用结果。

λ 列表示启用的每一项辅助 loss 权重：C3/E3/A3 两项分别取该值，不是二者之和。`C0.1` 为后续 bestfix baseline，在 portable configs 中映射到 EINV2 `C0`；不是历史缺失 best checkpoint 的旧 run。

## 1. 全部主矩阵结果

以下指标除末列外均来自独立 evaluation；末列是导出预测重评分的 validation SELD，不混入训练期在线 validation。

'''
    caveats = '''

## 2. 与同 seed baseline 的配对差

Δ = variant − baseline；比例差用百分点（pp）。这些是描述性配对差，不是显著性结论，也不把三个 seed 当三个独立数据集。

'''
    motion_intro = '''

## 3. 静态 / 动态定位诊断

**下表不是官方 LE_CD / LR_CD**：按 `(file, frame, class)` 做 Hungarian 角度匹配。frame LE 仅统计匹配成功项；matched recall 的分母是该组全部 GT source-frames；Recall@20 额外要求角误差不超过 20°。三 seed 均值 ± SD。

motion group 来自同 `(class, source)` 的连续 GT segment：若 segment 中存在相邻帧角位移 > 1e−6°，该 segment 的 source-frames 标记 dynamic，否则有至少两帧时为 static；不是逐瞬间速度阈值分组。Evaluation：static 64,036、dynamic 53,184，占 45.37%；不支持笼统称“动态样本很少”。Validation 对应 31,428 / 26,274。不同 class 分布可能不同，不据汇总推每类都改善。

'''
    ending = '''

## 4. 完成状态、缺口与历史边界

| 范围 | 已有证据 / 状态 |
|---|---|
| EINV2 C0/C1/C2/C3、E0/E1/E2/E3 | 每个 3 seeds，validation + evaluation 已有评分 |
| Multi C0/C1/C2/C3、A0/A3 | 每个 3 seeds；当前非零辅助权重 .05，validation + evaluation 已有评分 |
| Multi A1 / A2 | 有实现和 portable config；未进入本次统一结果矩阵，不能标记为已有三 seed 结果 |
| 历史 λ=.2、smoke/pilot | 不混入 λ=.05 主矩阵；本表不是所有历史试跑的穷尽目录 |
| 历史 EINV2 15.28° → 11.94° | single-seed/fold1、旧 metric；[历史证据](HISTORICAL_LE.md)独立保留，不替代本表 |
| E0 seed2026 | 归档 checkpoint 与预测可核查；最初启动/退出日志缺失，训练来源仅 PARTIALLY VERIFIED |
| 本批补齐 | 历史执行凭据：12 次补训练（8 EINV2、4 Multi）、77 项评测、2 项 motion 作业；不把 91 作业当 91 独立实验。RB05 本批准备队列未启动，不重复计数 |
| 模型版本与评分版本 | 历史 checkpoint 按旧 validation 规则选择，再统一重评分；不能声称已按新 metric 重训/重选。当前 portable config 是复现入口，不是原始 run config |

## 5. 核验结论与限制

当前可 **Share with caveats**：已从原始 run 行独立重算全部主口径均值、sample SD、配对差和 SELD 公式；本轮服务器只读核查 84 行 config/checkpoint/result 均存在，预测文件集合全部匹配参考。只检查 checkpoint 文件存在及大小，未反序列化或重新训练。配对诊断已确认使用同一预测目录。

模型/方法相关 62 个源文件中，54 个与仓库逐字节一致；8 个差异为已登记的 scalar 路径参数化（7）与 metric protocol 标识（1），不是架构/loss 改动。文件证据成立不等于独立复现或统计显著。

11/11 fallacy checks：

| 检查 | 本表处理 / 剩余限制 |
|---|---|
| Simpson's paradox | 全局与 motion 分层分开；未声称无 class-level 反转 |
| Ecological fallacy | run 均值不推断每个声源收益 |
| Berkson selection | LE 条件于匹配，须与 recall 同看 |
| Collider bias | 不对匹配子集作无偏总体定位因果解释 |
| Base rate neglect | 明列 static/dynamic GT 分母 |
| Regression to mean | 保留全部 3 seeds，不以最优 seed 宣称泛化 |
| Survivorship | A1/A2 缺口、E0 日志缺失、旧 best 缺失显式标记 |
| Look-elsewhere | 所有 14 主组展开；不报告筛选后“显著”结论 |
| Forking paths | 保留旧权重选择、不同 λ / metric / split 的界限 |
| Correlation / causation | 不据辅助 loss 比较排除架构、初始化、mask 等混杂 |
| Reverse causality | 非横断面因果研究，本项不适用 |

## Evidence Index

- [42 个逐 seed run](catalog_20260904/RUN_INDEX.md)；[84 行完整 config/checkpoint/result 路径](catalog_20260904/run_index.csv)。
- [原始 336 行评分](aligned_20260904/aligned_run_metrics.csv)、[112 组多协议汇总](aligned_20260904/aligned_aggregate_metrics.csv)、[原配对差](aligned_20260904/aligned_paired_seed_deltas.csv)：除主 micro 外还含 macro / 两种 legacy profile，不能横向混比。
- [本轮服务器证据检查](catalog_20260904/source_audit.json)、[本地重算检查](catalog_20260904/catalog_checks.json)。
- [168 行 motion 诊断](catalog_20260904/completion_motion_per_run.csv)、[分母](catalog_20260904/completion_motion_support.csv)。
- [12 次新增训练记录](catalog_20260904/completion_new_training.csv)、[历史作业 QA](catalog_20260904/completion_qa_checks.json)：仅作执行来源，不把其中旧 metric 列改名为新结果。
- [可复算生成脚本](../scripts/reports/build_experiment_catalog.py)、[只读源文件检查脚本](../scripts/reports/audit_result_sources.py)。
- [方法与架构图](../docs/METHOD_AND_ARCHITECTURE.md)。

复算：`python scripts/reports/build_experiment_catalog.py`。仅读取仓库快照并更新派生表，不访问 GPU、数据集或实验目录。
'''
    content = title + scope + table(['Model / Variant', 'Context', 'λ', 'Seeds', 'LE_CD ° ↓', 'LR_CD % ↑', 'F20 % ↑', 'ER20 ↓', 'SELD ↓', 'Val SELD ↓'], main_table)
    content += caveats + table(['Variant', 'Baseline', 'ΔLE °', 'ΔLR pp', 'ΔF20 pp', 'ΔSELD', 'LE↓ seeds', 'SELD↓ seeds'], delta_table)
    content += motion_intro + table(['Variant', 'Group', 'GT frames', 'frame LE ° ↓', 'matched recall % ↑', 'Recall@20 % ↑'], motion_table) + ending
    write_text(ROOT / 'reports/EXPERIMENT_CATALOG.md', content)
    checks = {'status': 'PASS', 'run_split_rows': len(runs), 'unique_runs': len(index) // 2, 'groups': len(ordered),
              'sample_sd_ddof': 1, 'recomputed_metric_groups': len(groups) * len(METRICS),
              'motion_rows': len(motion), 'checked_server_table_hashes': checked,
              'training_rerun': False, 'inference_rerun': False, 'statistical_significance_tested': False}
    write_text(OUT / 'catalog_checks.json', json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
