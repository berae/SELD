"""Descriptive paired reporting, fixed head seed2026; no refitting, selection or bootstrap."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

NAMES = {'F0':'原模型', 'F-EMA':'原模型＋两帧平滑', 'F-KF':'原模型＋卡尔曼滤波',
         'F-Deriv':'原模型＋独立运动预测头', 'R0':'原模型＋方向修正',
         'R1':'方向修正＋历史输入', 'R2':'方向修正＋历史输入＋运动监督'}
ORDER = ('F0','F-EMA','F-KF','F-Deriv','R0','R1','R2')
METRICS = ('ER20','F20','LE_CD','LR_CD','SELD_LR')
PAIRS = (('R0','F0'),('R0','F-Deriv'),('R1','R0'),('R2','R1'))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def main():
    p = argparse.ArgumentParser()
    for k in ('root','previous','output'):
        p.add_argument('--'+k, type=Path, required=True)
    a = p.parse_args()
    all_sealed = read(a.root/'ALL_EIGHT_FROZEN.json')
    assert all_sealed['status'] == 'PASS' and all_sealed['new_formal_runs'] == 8
    a.output.mkdir(parents=True, exist_ok=False)
    full = []; paired = []; training = []; sources = {}; diagnostics = []; denominators = []
    previous_gt = None; records = {}
    for seed in (2026,2027,2028):
        cohort = a.previous if seed == 2026 else a.root/f'C0_{seed}'
        ev = cohort/'evaluation'; frozen = cohort/'frozen'
        done = read(ev/'COMPLETED.json'); summary = read(ev/'SUMMARY.json'); seal = read(frozen/'FROZEN.json')
        assert done['status'] == 'PASS' and done['files'] == 200
        assert summary['freeze_sha256'] == sha(frozen/'FROZEN.json')
        assert not summary['new_blind_test'] and not summary['selection_on_evaluation']
        assert not summary['bootstrap_repeated']
        inp = read(cohort/'cache/evaluation/input_manifest.json')
        if previous_gt is None:
            previous_gt = inp['gt_files']
        assert inp['gt_files'] == previous_gt
        selected = [r for r in summary['results'] if r['head_seed'] in (None,2026)]
        assert len(selected) == 7 and {r['condition'] for r in selected} == set(ORDER)
        selected = {r['condition']:r for r in selected}; records[seed] = selected
        for c in ORDER:
            row = selected[c]
            assert row['C0_seed'] == seed and row['files'] == 200
            assert all(row['detection_preservation'].values())
            assert row['sed'] == selected['F0']['sed']
            if row['head_seed'] is not None:
                assert row['head_prefix_max_delta'] == 0
            assert set(row['prediction_sha256']) == set(previous_gt)
            for name, h in row['prediction_sha256'].items():
                assert sha(ev/row['id']/name) == h
            full.append(dict(baseline_seed=seed, name=NAMES[c], **row))
        for run in seal['runs']:
            if run['head_seed'] != 2026:
                continue
            result = read(frozen/Path(run['weight_path']).parent/'COMPLETED.json')
            assert result['status'] == 'PASS'
            assert result['best_epoch'] == run['selected_epoch'] and result['epochs'] == run['stopped_epoch']
            assert sha(frozen/run['weight_path']) == run['weight_sha256']
            assert selected[run['condition']]['weight_sha256'] == run['weight_sha256']
            training.append(dict(baseline_seed=seed, name=NAMES[run['condition']], reused=(seed==2026),
                condition=run['condition'], head_seed=2026, best_epoch=result['best_epoch'],
                stopped_epoch=result['epochs'], stopping_reason=result['stopping_reason'],
                convergence_confirmed=result['convergence_confirmed'], seconds=result['elapsed_seconds'],
                checkpoint_sha256=run['weight_sha256'], validation_scores=result['result']['scores']))
        for newer, reference in PAIRS:
            delta = {m:selected[newer]['scores']['dcase2023_micro'][m]-selected[reference]['scores']['dcase2023_micro'][m] for m in METRICS}
            paired.append(dict(baseline_seed=seed, comparison=NAMES[newer]+' − '+NAMES[reference],
                               newer=newer, reference=reference, delta=delta))
        diag = read(ev/'fixed_matching_diagnostics.json')
        ids = {r['id']:r['condition'] for r in selected.values()}
        wanted = [r for r in diag['summary'] if r['condition'] in ids]
        f0_den = {r['stratum']:r['count'] for r in wanted if r['condition']=='F0'}
        assert len(wanted) == 49
        for row in wanted:
            assert row['count'] == f0_den[row['stratum']]
            diagnostics.append(dict(baseline_seed=seed, name=NAMES[ids[row['condition']]], **row))
        denominators.append(dict(baseline_seed=seed, counts=f0_den,
            fixed_target_audit=diag['fixed_target_audit'], all_GT_source_coverage=diag['all_GT_source_coverage']))
        sources[str(seed)] = dict(evaluation_summary_sha256=sha(ev/'SUMMARY.json'),
            frozen_manifest_sha256=sha(frozen/'FROZEN.json'), diagnostics_sha256=sha(ev/'fixed_matching_diagnostics.json'),
            input_manifest_sha256=sha(cohort/'cache/evaluation/input_manifest.json'),
            C0_checkpoint_sha256=seal['C0_sha256'], path=str(cohort))
    aggregate = []
    for newer, reference in PAIRS:
        rows = [r for r in paired if r['newer']==newer and r['reference']==reference]
        assert len(rows) == 3 and {r['baseline_seed'] for r in rows} == {2026,2027,2028}
        aggregate.append(dict(comparison=rows[0]['comparison'], n_baselines=3,
            statistics={m:dict(mean=statistics.mean(r['delta'][m] for r in rows),
                               sample_SD=statistics.stdev(r['delta'][m] for r in rows)) for m in METRICS}))
    assert len(full)==21 and len(training)==12 and sum(not r['reused'] for r in training)==8
    save(a.output/'PAIRED_RESULTS.json',dict(official_results=full, per_baseline_paired_differences=paired,
        paired_mean_sample_SD=aggregate, training=training, fixed_matching_diagnostics=diagnostics,
        denominators=denominators, sources=sources, script_sha256=sha(Path(__file__)),
        independent_baselines=3, fixed_head_seed=2026, new_heads=8, reused_heads=4,
        original_head_seed_stability_separate=True, bootstrap_repeated=False, new_blind_test=False))
    lines=['# 三份baseline固定小头种子的配对结果', '', '## Material Passport', '',
        '- 2026-09-09；academic-research-suite / experiment-agent；run＋descriptive validation。',
        '- 3个baseline，head seed均2026；复用原baseline4头，新增两份baseline8头。不是12个独立baseline。',
        '- 全部8个新增头先冻结再evaluation；每条件200条。历史evaluation已查看，不是新盲测。',
        '- 下列为DCASE2023 micro；JSON保留全部官方协议、逐baseline差及来源。无bootstrap或显著性检验。',
        '', '## 全部官方结果', '', table(['baseline','条件',*METRICS],
            [[r['baseline_seed'],r['name'],*[f"{r['scores']['dcase2023_micro'][m]:.6f}" for m in METRICS]] for r in full]),
        '', '## 逐baseline配对差（前者减后者）', '',
        'ER、LE、SELD负值较好；F20、LR正值较好。所有组保留，不按evaluation选seed。', '',
        table(['baseline','比较',*METRICS],[[r['baseline_seed'],r['comparison'],*[f"{r['delta'][m]:+.6f}" for m in METRICS]] for r in paired]),
        '', '## 配对差的三baseline等权均值 ± sample SD', '',
        table(['比较',*METRICS],[[r['comparison'],*[f"{r['statistics'][m]['mean']:+.6f} ± {r['statistics'][m]['sample_SD']:.6f}" for m in METRICS]] for r in aggregate]),
        '', 'SD描述这三个权重之间的离散程度，不是置信区间，也不据此宣称统计显著。',
        '', '## 停止与选优（两者分开）', '',
        table(['baseline','条件','复用/新增','停止轮','最佳轮','双平台确认','秒'],
              [[r['baseline_seed'],r['name'],'复用' if r['reused'] else '新增',r['stopped_epoch'],r['best_epoch'],
                '是' if r['convergence_confirmed'] else '否',f"{r['seconds']:.1f}"] for r in training]),
        '', '完整停止原因、validation分数、权重hash在JSON；达到200轮而未双平台者不得写成已确认收敛。',
        '', '## 检测保持与固定分母', '',
        '21条件均通过有序检测条目、纯SED计数和原始概率保持检查；学习头所测因果前缀差为0。空间F20/定位召回允许因方向变化而改变。', '',
        table(['baseline',*denominators[0]['counts'].keys()],
              [[d['baseline_seed'],*d['counts'].values()] for d in denominators]),
        '', '分母在同baseline各方法间固定，不在不同baseline间强行相等；无20°筛样，无重新匹配。pair分层与整段source分层分开。完整静态/动态误差、漏检和未匹配计数见JSON。',
        '', '## 结论边界', '',
        '- 原固定C0的三头seed稳定性单列，沿用既有STABILITY_RESULTS与EVALUATION_RESULTS，不重算或混计。',
        '- 仅一个数据集、同一主干、三个既有baseline权重；不支持跨架构或跨数据集泛化。',
        '- 历史输入同时改变参数量，缺少等容量对照；不能把增量结果解释成历史信息本身无用。',
        '- 预检修正仅取同一训练录音更多chunk；旧失败、授权、缓存差异和新检查均保留。',
        '', '## 来源', '', table(['baseline','evaluation SUMMARY SHA256','FROZEN SHA256'],
             [[seed,s['evaluation_summary_sha256'],s['frozen_manifest_sha256']] for seed,s in sources.items()])]
    (a.output/'PAIRED_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',baselines=3,conditions=21,new_heads=8,bootstrap_repeated=False)))


if __name__=='__main__':
    main()
