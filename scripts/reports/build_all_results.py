"""Rebuild the complete, version-separated experiment ledger from saved evidence.

Standard library only. No training, inference, network calls or source-run writes.
"""
import csv
import io
import json
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/summary_20260907'
METRICS = ('ER20', 'F20', 'LE_CD', 'LR_CD', 'SELD_LR')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(str(x).replace('|', '/') for x in r) + ' |' for r in rows])


def modern_label(run):
    cfg = run['config']; train = cfg['training']; variant = cfg['audit']['variant']
    if train.get('velocity_min_norm', 0):
        return 'D1_motionpairs' if variant == 'C1' else 'D3_motionpairs_j005'
    if variant == 'C3':
        return 'C3_j005' if train['lambda_jepa'] == .05 else 'C3_j020'
    return variant


def main():
    evidence = json.loads((OUT / 'evidence.json').read_text(encoding='utf-8'))
    if evidence.get('format') == 'file_references_v1':
        for host in evidence['hosts']:
            for key in ('runs','legacy_validation','manifests','legacy_tables'):
                host[key] = [json.loads((OUT / p).read_text(encoding='utf-8')) for p in host[key]]
    rows, smoke = [], []
    old = read_csv(ROOT / 'reports/catalog_20260904/run_index.csv')
    for r in old:
        rows.append(dict(cohort='historical_' + r['family'] + '_' + r['context'], host='rabbit02',
                         family=r['family'], variant=r['variant'], seed=int(r['seed']), split=r['split'],
                         dataset=r['dataset'], train_folds=r['train_folds'], validation_fold=r['validation_fold'],
                         run_id=r['run_id'], training_status='HISTORICAL_CHECKPOINT_PRESENT',
                         scoring_status='SCORED_SAVED_PREDICTIONS', evidence='VERIFIED_ARTIFACTS',
                         config_path=r['historical_config'], checkpoint_path=r['checkpoint'],
                         result_path=r['original_result'], prediction_path=r['prediction_dir'],
                         profile=r['profile'], checkpoint_bytes=int(r['checkpoint_bytes']),
                         **{k:float(r[k]) for k in METRICS}))
    for host in evidence['hosts']:
        for run in host['runs']:
            cfg=run['config']; audit=cfg['audit']; status=run['status']; path=PurePosixPath(run['run_dir'])
            if audit['smoke_batches']:
                smoke.append(dict(host=host['host'],run_dir=str(path),status=status.get('status'),
                                  epoch=status.get('epoch'),error=status.get('error',''),
                                  config_path=str(path/'config.json'),checkpoint_present=run['files']['best.pth']['exists']))
                continue
            for split in ('validation','evaluation'):
                result=run['metrics'].get(split)
                rows.append(dict(cohort='audited_'+host['host'],host=host['host'],family='EINV2',
                                 variant=modern_label(run),seed=audit['seed'],split=split,dataset='TAU2020 FOA',
                                 train_folds=cfg['training']['train_fold'],validation_fold=cfg['training']['valid_fold'],
                                 run_id=path.name,training_status=status.get('status','UNKNOWN'),
                                 scoring_status='COMPLETED' if result else 'NOT_FOUND',evidence='VERIFIED' if result else 'PARTIALLY VERIFIED',
                                 source_version=audit['version'],lambda_velocity=cfg['training']['lambda_velocity'],
                                 lambda_jepa=cfg['training']['lambda_jepa'],velocity_min_norm=cfg['training'].get('velocity_min_norm',0),
                                 epoch=status.get('epoch'),best_epoch=status.get('best',{}).get('epoch'),
                                 config_path=str(path/'config.json'),checkpoint_path=str(path/'best.pth'),
                                 result_path=str(path/split/'metrics.json'),prediction_path=str(path/split),
                                 profile='dcase2023_micro',checkpoint_bytes=run['files']['best.pth'].get('bytes',0),
                                 **(result['scores']['dcase2023_micro'] if result else {})))
    rows.sort(key=lambda r:(r['cohort'],r['variant'],r['seed'],r['split']))
    assert len(rows)==len({(r['cohort'],r['run_id'],r['split']) for r in rows})
    checks=[]
    for r in rows:
        if 'SELD_LR' in r:
            expected=(r['ER20']+1-r['F20']+r['LE_CD']/180+1-r['LR_CD'])/4
            assert abs(expected-r['SELD_LR'])<1e-9,r
    for host in evidence['hosts']:
        for r in host['runs']:
            if not r['config']['audit']['smoke_batches']:
                assert r['status']['status']=='completed' and r['status']['epoch']==90
                assert r['log_rows']==90 and r['files']['best.pth']['bytes']>0 and r['files']['latest.pth']['bytes']>0
                for s,m in r['metrics'].items():
                    assert m['files']==(100 if s=='validation' else 200)
                    if s=='validation':assert m.get('training_validation_max_abs_diff',0)<1e-6
        checks.extend(host['legacy_validation'])
    assert all(x['exists'] for r in checks for x in r['checks'].values())
    write_csv('runs.csv',rows);write_csv('smoke_and_failures.csv',smoke)
    groups=defaultdict(list)
    for r in rows:groups[(r['cohort'],r['variant'],r['split'])].append(r)
    aggregates=[]
    for (cohort,variant,split),group in sorted(groups.items()):
        scored=[r for r in group if 'SELD_LR' in r]
        row=dict(cohort=cohort,variant=variant,split=split,n_runs=len(group),n_scored=len(scored),
                 seeds=','.join(str(r['seed']) for r in group),dataset='TAU2020 FOA',profile='dcase2023_micro')
        for m in METRICS:
            values=[r[m] for r in scored]
            row[m+'_mean']=statistics.mean(values) if values else ''
            row[m+'_sd']=statistics.stdev(values) if len(values)>1 else ''
        aggregates.append(row)
    write_csv('aggregate.csv',aggregates)
    paired=[]
    baselines={'audited_rabbit02':'C0','audited_RB05':'C0','historical_EINV2_causal':'C0.1',
               'historical_EINV2_noncausal':'E0','historical_Multi-ACCDOA_causal':'C0','historical_Multi-ACCDOA_noncausal':'A0'}
    for r in rows:
        base=baselines.get(r['cohort'])
        if not base or r['variant']==base or 'SELD_LR' not in r:continue
        b=next((b for b in rows if b['cohort']==r['cohort'] and b['variant']==base and b['seed']==r['seed'] and b['split']==r['split']),None)
        if b is None:continue
        paired.append(dict(cohort=r['cohort'],variant=r['variant'],baseline=base,seed=r['seed'],split=r['split'],
                           **{m+'_delta':r[m]-b[m] for m in METRICS}))
    write_csv('paired_seed_deltas.csv',paired)
    # Include every captured manifest, not only the chosen main-matrix variants.
    registry=[]
    original_multi=[]
    for host in evidence['hosts']:
        for item in host['manifests']:
            p=item['path'];m=item['data']
            if '/DynamicCausalMultiACCDOA_TAU2020/runs/manifests/' in p:
                params=m.get('params',{})
                for split,values in [('validation',m.get('best_validation',{})),('evaluation',m.get('evaluation',{}))]:
                    if values:
                        original_multi.append(dict(run_id=m.get('job_id',''),variant=m.get('variant',''),seed=m.get('seed'),
                                                   lambda_velocity=params.get('lambda_velocity',0),lambda_jepa=params.get('lambda_jepa',0),
                                                   split=split,status=m.get('status'),manifest_path=p,
                                                   metric_protocol='original_manifest_NOT_current_aligned',
                                                   **{k:(v[0] if isinstance(v,list) else v) for k,v in values.items() if k in ('ER','F','LE','LR','SELD')}))
            registry.append(dict(host=host['host'],record_type='manifest',path=p,run_id=m.get('job_id',m.get('unique_name','')),
                                 variant=m.get('variant',''),seed=m.get('seed',''),status=m.get('status','UNKNOWN'),
                                 checkpoint=m.get('checkpoint',''),prediction_dir=m.get('prediction_dir',''),
                                 original_metric_protocol='as_recorded_NOT_relabelled',
                                 best_validation_json=json.dumps(m.get('best_validation',{}),ensure_ascii=False),
                                 evaluation_json=json.dumps(m.get('evaluation',{}),ensure_ascii=False)))
        for item in host['legacy_tables']:
            if 'text' not in item:continue
            text=item['text'];delimiter='\t' if '\t' in text.splitlines()[0] else ','
            data=list(csv.DictReader(io.StringIO(text),delimiter=delimiter))
            row=dict(host=host['host'],record_type='training_curve' if item['path'].endswith('metrics_statistics.csv') else 'evaluation_table',
                     path=item['path'],n_logged_rows=len(data),original_metric_protocol='as_recorded_NOT_relabelled',
                     status='LOGGED_METRICS_ONLY' if data else 'HEADER_ONLY_NO_METRICS',last_row_json=json.dumps(data[-1] if data else {},ensure_ascii=False))
            key=next((k for k in ('SELD_scr_macro','seld20','SELD_scr','SELD') if data and k in data[0]),None)
            if key:
                best=min(data,key=lambda x:float(x[key]));row['minimum_logged_selection_metric']=key
                row['minimum_logged_row_json']=json.dumps(best,ensure_ascii=False)
            registry.append(row)
    write_csv('legacy_registry.csv',registry)
    write_csv('original_multi_results.csv',original_multi)
    old_groups=defaultdict(list)
    for r in original_multi:
        if r['run_id'].startswith('paper_v1_') and r['split']=='evaluation':
            old_groups[(r['variant'],r['lambda_velocity'],r['lambda_jepa'])].append(r)
    original_table=table(['旧Multi Variant','λ velocity / JEPA','n','LE 原始','LR 原始 %','F 原始 %','SELD 原始'],
                         [[variant,f'{lv}/{lj}',len(a),*[
                             f'{statistics.mean(x[m] for x in a)*scale:.3f} ± {statistics.stdev(x[m] for x in a)*scale:.3f}'
                             for m,scale in [('LE',1),('LR',100),('F',100),('SELD',1)]]]
                          for (variant,lv,lj),a in sorted(old_groups.items())])
    summary=dict(formal_main_runs=len(rows)//2,scored_run_splits=sum('SELD_LR' in r for r in rows),
                 formal_new_runs=sum(r['cohort'].startswith('audited_') for r in rows)//2,
                 missing_evaluation=[r['run_id'] for r in rows if r['split']=='evaluation' and 'SELD_LR' not in r],
                 smoke_runs=len(smoke),failed_smokes=sum(r['status']=='failed' for r in smoke),
                 legacy_records=len(registry),formula_checks='PASS',legacy_path_checks='PASS',
                 modern_epoch_log_checkpoint_checks='PASS',checkpoint_payload_deserialized=False,
                 scope='two project roots; current+historical summaries, no dataset/venv/symlink traversal; no independent retraining')
    (OUT/'qa.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def fmt(r,m,scale=1):
        mean=r[m+'_mean'];sd=r[m+'_sd']
        if mean=='':return '—'
        return f'{mean*scale:.3f}'+(f' ± {sd*scale:.3f}' if sd!='' else ' (n=1)')
    dashboard=[]
    for a in aggregates:
        if a['split']!='evaluation':continue
        r=next(r for r in rows if r['cohort']==a['cohort'] and r['variant']==a['variant'] and r['split']=='evaluation')
        dashboard.append({'Experiment':a['cohort'],'Variant':a['variant'],
                          'Status':'evaluation完成' if a['n_scored']==a['n_runs'] else '训练完成/evaluation缺失',
                          'Dataset':a['dataset'],'Seed/Fold':a['seeds']+' / val fold1',
                          'Key Metrics':'LE_CD / LR_CD% / F20%',
                          'Result':fmt(a,'LE_CD')+' / '+fmt(a,'LR_CD',100)+' / '+fmt(a,'F20',100),
                          'Evidence Path':r['result_path']+' ; all seeds: runs.csv',
                          'Next Action':'补独立evaluation' if not a['n_scored'] else '补新版motion分层' if a['cohort'].startswith('audited_') else '保留历史分组，不混新版'})
    write_csv('dashboard.csv',dashboard)
    (OUT/'DASHBOARD.md').write_text('# Experiment Dashboard\n\n'+table(list(dashboard[0]),[list(r.values()) for r in dashboard])+'\n',encoding='utf-8')
    sections=['# SELD 全部实验结果与代码整理（2026-09-07）',
              '证据来自 rabbit02 `/work/zhanghc/Myllm/SELD` 与 RB05 `/home/zhanghc/SELD` 的实际文件；不采用聊天中的成绩作为数据源。',
              '[Dashboard](summary_20260907/DASHBOARD.md) · [代码地图](../docs/CODE_MAP.md)',
              '## 1. 总结',
              f"主结果矩阵 **{summary['formal_main_runs']} 个 run**（历史42 + 重审/补充26），**{summary['scored_run_splits']} 个已评分 run×split**；新版26个正式run均完成90 epochs，21个完成独立evaluation，5个仅validation。另列 {len(smoke)} 个新版smoke/初始化记录和 {len(registry)} 条旧版manifest/训练曲线/评估记录；后者包含重复评测和迁移副本，不能相加当独立实验数。",
              'VERIFIED：新版 velocity-only 在两台机器上都呈现平均 LE 改善，但 LR 下降。低权重 JEPA 联合方案比高权重方案 validation 更好；尚未证明优于 velocity-only 或稳定互补。不能概括为所有 EINV2 成功，亦不能说 Multi-ACCDOA 完全无定位收益。',
              '## 2. 研究目标与实际方法',
              '目标：以显式 DOA 一阶差分 velocity 监督与隐空间未来预测互补，提高定位 LE，尽量维持 LR/F20。EINV2 与 Multi-ACCDOA 均有 velocity head、latent projector/predictor、EMA teacher 和辅助 loss；不把已有 DOA derivative 思路当全新贡献。独立 acceleration 监督并非本次主矩阵方法。',
              '新版 EINV2 采用同一 C3 架构做 C0/C1/C2/C3 配对（零权重关闭梯度）；PIT 对齐后计算 velocity 与 teacher/online 轨道 latent。JEPA 是训练期增添的结构，不只是标量 loss；当前推理仍会计算部分辅助输出，不能声称已裁剪到零推理开销。',
              '## 3. 统一评价口径与设置',
              '主表采用固定 DCASE2023 official core + micro；这是 metric 版本，不是 DCASE2023 数据。F20 为 location-dependent F，不是纯 SED F1；未收集 mAP。旧 LR20 可能是 localization F，原始表保留原名，不冒充 LR_CD。',
              'TAU2020 FOA；train folds2–6、validation fold1、evaluation独立200条录音；seeds 2026/2027/2028。新版 EINV2：24 kHz、FFT1024/hop600、256 mel + intensity，4 s独立chunk，100 ms label内允许75 ms依赖；不是零延迟/跨chunk流式缓存。train-only scaler、FOA reorder=False、batch32、Adam amsgrad lr5e-4、epoch80后×0.1、固定90 epochs，按 validation SELD_LR 最小选best，**不是90前早停**。',
              'velocity λ=.2，JEPA λ=.2或.05；预测 horizons=[1,3,5] frames，latent128、predictor hidden256、EMA=.996。D1/D3仅对 target velocity norm>1e-6 的有效pair施加velocity loss。完整逐run配置与源码hash见 evidence.json；旧版与新版前端/选优口径不同，不合并均值。Multi旧主矩阵非零λ=.05，模型/训练配置见历史catalog。',
              '## 4. 全部主结果：evaluation',
              'mean ± sample SD；LE为度，LR/F20为百分比。n=1不填写SD；缺少evaluation不填零。每一行按 cohort/variant 在 [runs.csv](summary_20260907/runs.csv) 可找到全部config/checkpoint/result绝对路径。']
    def aggregate_table(split):
        return table(['Cohort','Variant','n scored/runs','LE_CD ↓','LR_CD % ↑','F20 % ↑','ER20 ↓','SELD_LR ↓'],
                     [[r['cohort'],r['variant'],f"{r['n_scored']}/{r['n_runs']}",fmt(r,'LE_CD'),fmt(r,'LR_CD',100),fmt(r,'F20',100),fmt(r,'ER20'),fmt(r,'SELD_LR')] for r in aggregates if r['split']==split])
    sections += [aggregate_table('evaluation'),'## 5. Validation 与缺失实验',aggregate_table('validation'),
                 'RB05 C3_j020三seeds、D1/D3 seed2026没有独立evaluation文件；不是failed test。D1/D3各有一个90epoch validation结果，但未扩到三seeds，不能据整体validation断言动态子集收益。新版smoke有2次初始化失败（错误HDF5路径），修正路径后的smoke成功，正式训练成功；保留失败记录。',
                 '## 6. 静态/动态与消融状态',
                 '历史42主run已有static/dynamic诊断：[完整分层表](EXPERIMENT_CATALOG.md#3-静态动态定位诊断)及[168条原始分层记录](catalog_20260904/completion_motion_per_run.csv)。eval GT source-frame static=64036、dynamic=53184（45.37%）；这是该分层定义下的比例，不等于moving pair比例，不能仅以“动态很少”解释性能下降。分层Recall@20是自定义逐帧指标，不是官方LR_CD。',
                 '新版EINV2的motion分层结果本次未找到；因此旧版分层收益不能直接移植到新版。新版 rabbit02 C0–C3三seed完整；RB05有C0/C1/联合两种JEPA权重，缺少JEPA-only λ=.05三seed；D1/D3仅seed2026。**更正历史概括：Multi A1/A2在旧λ=.2矩阵已有三seed，不能说只有代码；缺的是λ=.05三seed与统一重评分。**',
                 '## 7. 早期、旁支与失败记录（不删除）',
                 '[legacy_registry.csv](summary_20260907/legacy_registry.csv)列出全部检索到的旧manifest、原始评估表、训练curve的末行及最低logged selection metric行。原始数值见evidence.json；不将log最小行自动当作保留checkpoint，也不对不同metric作均值。包含旧Multi λ=.2、smoke、早期baseline，RB05旧EINV2 λ=.05，以及STARSS22/23。',
                 '### 7.1 旧Multi完整8×3矩阵（原始manifest口径）',
                 '下表不是重新对齐后的DCASE主表，故不与第4节数值直接做差。C0/A0同一run也出现在统一表中，不重复计数；列表式原始评价值中的第二项是该run内部区间，下面SD重新按三个seed计算。',original_table,
                 '其余λ=.02 pilot、λ=.05、smoke的逐run validation/evaluation数值都在[original_multi_results.csv](summary_20260907/original_multi_results.csv)。',
                 '### 7.2 STARSS训练曲线（非独立evaluation）',
                 '| Dataset / seed | logged epochs | 最低logged macro-SELD所在epoch | LE_macro | LR_macro % | F_macro % | SELD_macro |\n|---|---|---|---|---|---|---|\n| STARSS22 / 2026 | 80 | 64 | 42.863 | 47.989 | 21.246 | 0.600075 |\n| STARSS23 / 2026 | 80 | 78 | 30.934 | 43.907 | 20.605 | 0.578386 |',
                 'STARSS22/23实际有training curve，也有仅header的启动记录；未找到本次可确认的独立test结果，不能声称STARSS baseline已完整复现。ObjectStateSELD有B0工程smoke与manifest/配置，但README明确完整训练/评价及B1–B3 objectives未完成，本轮未发现正式训练成绩。PSELDNets/NTU等上游目录单列路径与Git记录，不算本方法结果。',
                 '## 8. 证据强度与设计问题',
                 'VERIFIED：主表评分文件、逐run配置、checkpoint存在/大小、90条epoch日志与完成状态、评价文件数、SELD公式重算。PARTIALLY VERIFIED：checkpoint未逐一反序列化或独立重训；旧E0 seed2026启动来源不完整；STARSS仅训练曲线。PLANNED：缺失evaluation/三seed/motion分层/JEPA-only .05。INFERRED：显式运动与latent连续性互补的机制解释，尚无稳定消融证明。',
                 '风险：LE条件于检测匹配，LE改善同时LR下降不能直接解释为全体声源定位提升；λ搜索、mask试验与多seed比较是探索性结果。每组仅3seeds/一个validation fold，不报告显著性或广泛泛化。rabbit02与RB05不混为6seeds；相同seed跨硬件不保证逐位一致。历史checkpoint按旧指标选优后重评分，与新版不同。未来teacher目标用于训练不自动构成推理泄漏；因果声明仍以当前帧边界与端到端依赖审计为准。',
                 '## 9. 下一步（本轮只整理，未启动）',
                 '1. P0：冻结新版各组best及当前表，补新版同一预测上的static/dynamic LE、匹配分母、recall，检验定位收益是否来自漏检变化。\n2. P0：补齐RB05高JEPA与D1/D3现有checkpoint的独立evaluation，结果作为探索性附录，不用test继续调λ。\n3. P1：若继续验证互补性，补RB05 JEPA-only λ=.05三seeds，与B0/B1/联合形成完整2×2。\n4. P1：以velocity-only为主定位对照，预先固定LE/LR/F20可接受权衡；D1/D3当前无充分推广依据，先不扩展新的loss。',
                 '## 10. 代码与版本',
                 '当前代码、训练入口、评价入口、配置、服务器路径预设、历史source snapshots分层管理，见[代码地图](../docs/CODE_MAP.md)。服务器原目录不重命名、不删除、不修改；权重/数据/完整预测不上传。三个新版runtime按训练记录的逐文件SHA256恢复，防止用当前源码错评旧checkpoint。',
                 '## Evidence Index',
                 '- [全部68主run×split与绝对路径](summary_20260907/runs.csv)、[均值/SD](summary_20260907/aggregate.csv)、[配对seed差](summary_20260907/paired_seed_deltas.csv)。\n- [原始配置/评分/状态/曲线/manifest快照](summary_20260907/evidence.json)、[核验结果](summary_20260907/qa.json)、[smoke/失败](summary_20260907/smoke_and_failures.csv)、[旧记录索引](summary_20260907/legacy_registry.csv)。\n- [服务器目录与Git索引](summary_20260907/server_inventory.json)、[checkpoint索引](summary_20260907/checkpoints.csv)。\n- [训练期runtime源文件快照](../provenance/runtime_snapshots)、[新增旧项目代码来源](../provenance/export_rabbit02_20260907.json)。\n- [旧14组完整报告](EXPERIMENT_CATALOG.md)、[历史15.28→11.94的限定条件](HISTORICAL_LE.md)。',
                 '## ARS Material Passport / fallacy scan',
                 '材料：两台服务器本地实验文件，只读采集；摘要/计算由Codex生成，未经作者逐项人工确认；无USER_ATTESTED_READ声明。执行ARS experiment-agent validate，判断为Share with caveats，而非机制或论文结论已验证。',
                 '11项检查：Simpson分层待补；ecological不推断个体；Berkson/collider明确LE匹配选择；base-rate列分母；regression-to-mean保留全部seed；survivorship列失败/缺测；look-elsewhere不报筛选后显著性；forking-paths保留版本/λ；correlation≠causation不声称互补机制成立；reverse-causality不适用。']
    (ROOT/'reports/ALL_EXPERIMENTS.md').write_text('\n\n'.join(sections)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
