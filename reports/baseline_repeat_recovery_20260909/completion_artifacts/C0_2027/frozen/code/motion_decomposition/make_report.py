"""Render Chinese report from completed small share packages."""
import csv
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1]);rb=root/'RB05';rabbit=root/'rabbit02'
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
metrics=read(rb/'metrics_per_run.csv');common=read(rb/'common_match_breakdown.csv');pairs=read(rb/'paired_deltas.csv')
summary=read(rb/'seed_summary.csv')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])
def fmt(v,d=3):return f'{float(v):.{d}f}'
selected=[r for r in common if r['baseline']=='C0/raw' and r['variant']=='C1/raw' and r['split']=='evaluation' and r['stratum']=='all']
q1=table(['seed','全GT','双方匹配','仅C0（lost）','仅C1（new）','双方未匹配','共同目标 ΔLE°'],[[r['seed'],r['gt_denominator'],r['both_matched'],r['baseline_only'],r['variant_only'],r['neither'],fmt(r['common_delta_LE'])] for r in selected])
primary=[r for r in metrics if r['split']=='evaluation' and r['profile']=='dcase2023_micro' and r['variant'] in ('C0','C1','C3_j005')]
official=table(['模型/解码','LE° mean±SD','LR % mean±SD','F20 % mean±SD','ER20 mean±SD','SELD mean±SD'],[
    [v+'/'+d]+[f'{statistics.mean(float(r[k])*scale for r in primary if r["variant"]==v and r["decoder"]==d):.{digits}f} ± {statistics.stdev(float(r[k])*scale for r in primary if r["variant"]==v and r["decoder"]==d):.{digits}f}' for k,scale,digits in [('LE_CD',1,3),('LR_CD',100,3),('F20',100,3),('ER20',1,4),('SELD_LR',1,4)]]
    for v,d in [('C0','raw'),('C0','smoothing'),('C1','raw'),('C1','smoothing'),('C1','fusion'),('C3_j005','raw'),('C3_j005','smoothing'),('C3_j005','fusion')]])
p=[r for r in pairs if r['baseline']=='C1/smoothing' and r['variant']=='C1/fusion' and r['split']=='evaluation' and r['profile']=='dcase2023_micro']
q2=table(['seed','fusion−smoothing ΔLE°','ΔLR','ΔSELD'],[[r['seed'],fmt(r['delta_LE_CD'],6),fmt(r['delta_LR_CD'],7),fmt(r['delta_SELD_LR'],7)] for r in p])
q3=table(['同一解码器','C3−C1 ΔLE° mean±SD'],[[d,f'{r["mean"]} ± {r["sample_SD"]}'] for d in ('raw','smoothing','fusion') for r in summary if r['table']=='paired' and r['baseline']=='C1/'+d and r['variant']=='C3_j005/'+d and r['split']=='evaluation' and r['profile']=='dcase2023_micro' and r['metric']=='delta_LE_CD'])
appendix=[r for r in metrics if r['variant'] not in ('C0','C1','C3_j005') and r['profile']=='dcase2023_micro']
app=table(['模型','seed','LE°','LR %','F20 %','SELD'],[[r['variant'],r['seed'],fmt(r['LE_CD']),fmt(float(r['LR_CD'])*100),fmt(float(r['F20'])*100),fmt(r['SELD_LR'],4)] for r in appendix])
text=f'''# 固定 checkpoint 的运动作用拆分：执行结果

## Q1—Q3 的回答

**Q1：VERIFIED。C1 的收益并非完全来自“丢掉难目标”，但并不稳定。** RB05 evaluation 的原始官方 LE 配对差为 −0.917±1.536°；共同匹配目标上为 −0.305±1.115°（mean±sample SD，3 seeds）。seed2026/2027 共同目标改善，seed2028 变差；三个 seed 均 lost>new。必须同时看到共同目标收益和覆盖率损失，不能把平均 LE 改善概括为普遍改善。validation 共同目标变化依次为 −2.595、+0.321、+1.248°，同样不稳定。

{q1}

**Q2：VERIFIED。在这套固定关联、alpha=0.5、dt=0.1、非递归两帧设置中，没有证明预测速度比普通平滑提供稳定额外定位价值。** C1 smoothing 相对 raw 的平均 LE 改善0.0477°；fusion 相对 smoothing 的配对 ΔLE 为 +0.000135±0.024001°，几乎相同，SELD 平均略差。该结论仅覆盖本次已冻结条件，不是证明速度信息在所有解码器上都无效。

{q2}

**Q3：VERIFIED。C3_j005 在三个相同解码条件下均未显示优于 C1 的平均 LE。** 均值差约+0.33°且seed差异大，暂不将 JEPA 定为论文的额外定位贡献。

{q3}

## 主结果：RB05 evaluation

固定 DCASE2023 official core + micro，完整60秒；下面均为3 seeds的均值与sample SD。macro与逐seed精确值见CSV，不能混合两种平均。

{official}

主目录 [逐seed结果](RB05/metrics_per_run.csv)、[配对差](RB05/paired_deltas.csv)、[共同匹配分母与lost/new误差](RB05/common_match_breakdown.csv)、[静态/动态/重叠/起止/转弯诊断](RB05/motion_strata.csv)。纯SED的micro/macro F1及TP/FP/FN同列于逐seed结果；所有平滑/融合条件的纯SED逐元素不变。官方LR可有微小变化，自定义matched recall不是官方LR。

evaluation的全GT source-frame分母为117,220，其中static 64,036、dynamic 53,184；same-class overlap 4,334、different-class overlap 51,238、single 61,648。起止附近12,329、turn 7,422；jump仅3个GT，不能据此推广突变表现。validation全GT为57,702。分层matched数、Recall@20、误差和共同目标分解均在CSV，完整逐GT匹配表留服务器。

## 机制证据与限制

**VERIFIED：预测速度并非随机无信息。** C1 evaluation 的moving-pair分母每seed为49,155；预测速度向量L2误差为0.251/0.240/0.260，均低于零速度0.342，且低于raw DOA差分约0.885/0.936/0.937。GT速度范数中位数0.337，预测仅0.118/0.116/0.096。static_pair与全部pair也独立报告。这是离线GT/PIT对齐的速度质量证据，不能替代主解码效果。

**INFERRED：速度幅度偏小、静态pair的非零输出、以及主解码中的活动/关联条件可能限制增益；本轮没有把这些解释当作已证明的因果机制。**

**VERIFIED：边界监督现象存在。** 500条训练录音中290,722个有效velocity target有7,222个位于非首chunk起点，占2.484%；其中3,268个为运动target。源码直接截取整录音mask，首帧可能引用块外上一帧。没有修改监督、没有重训，不能声称旧成绩已修复，也不能仅凭这项统计把它认定为失败原因。

**VERIFIED：C3_j005 的latent预测没有稳定超越persistence。** evaluation 的100/300/500ms有效pair数分别112,128/102,287/93,305；三个seed在100和300ms的predictor误差均高于teacher persistence，500ms仅seed2026较低。online persistence另列。方差、有效秩和跨录音配对cosine均保留，不由LayerNorm或loss大小推断未坍缩或动态成功。rabbit02的C2 validation诊断另列，不与RB05混作配对实验。

见 [速度](RB05/velocity_diagnostics.csv)、[latent](RB05/latent_diagnostics.csv)、[latent离散度](RB05/latent_dispersion.json)、[训练chunk边界](RB05/data_audit/train_chunk_boundary.json)、[预定与失败案例](RB05/cases/case_selection.csv)。图中缺失值表示未匹配，不当作0°。

## 复核与探索性附录

rabbit02的C0/C1三seed、两个split已从原CSV独立重评分，均与保存评分一致；evaluation共同目标ΔLE为+0.522/−0.061/−1.483°。结果在 [rabbit02独立表](rabbit02/metrics_per_run.csv) 和 [共同目标表](rabbit02/common_match_breakdown.csv)，不能与RB05合并成六seed。这里没有重新导出rabbit02 C0/C1浮点结果或运行其融合矩阵。

RB05现场确认缺失后，补齐以下5个既有checkpoint的独立evaluation，仅作探索性附录，不据此调lambda或选择下一批搜索：

{app}

## 执行、QA与缺证据

- **VERIFIED**：主矩阵18个run×split浮点导出、48个解码评估；附录5个evaluation；11组单元测试通过；全部主raw回归与历史CSV/评分一致。首次失败导出排除在主结果外，修正版本独立保存。
- **VERIFIED**：RB05全部14个正式run、rabbit02全部12个正式run均读取了实时config/checkpoint哈希及runtime source_hashes；没有用聊天成绩代替输入。RB05 waveform/label输入另作哈希冻结。
- **VERIFIED**：历史完整性复核RB05 4,474项、rabbit02 4,284项均未改变。主解码没有GT身份、GT活动或未来帧输入；GT只用于评分诊断。
- **NOT RUN**：任何新训练、lambda扫描、梯度干预、GT oracle融合、recording bootstrap与新的独立盲测；梯度范数/夹角诊断也未执行。当前证据足以否定“已证明稳定互补”的表述，但不足以定位唯一训练原因。
- **来源限制**：本地代码从已有发布快照建立独立Git分支；上游51096ff的存在已网页核验，整树逐文件对应未验证。服务器冻结runtime按checkpoint哈希全量验证通过，见manifest。没有push main。

细节见 [方法与边界](METHODS.md)、[失败记录](FAILURES.md)、[版本manifest](input_manifest.json)、[冻结协议](RB05/execution/protocol.json)、[测试](RB05/execution/tests.json)、[最终QA](RB05/execution/final_qa.json)。完整权重、音频、浮点预测、逐GT匹配记录留在两台服务器各自的 `reports/motion_decomposition_20260907/`。

## 下一步决策

**暂不进入新的90-epoch训练，也不扩大JEPA或lambda搜索。** 唯一建议的最小后续实验是：固定C1 seed2026 checkpoint与validation，做独立标记的“GT位移oracle融合上界”诊断，保持同一个预测关联与活动规则，检查准确速度在该解码条件下是否确实有可利用的上界。它不进入主结果、不用于部署、不使用evaluation调参；若连上界也没有收益，就无需为这个融合规则投入新训练。本轮未执行该oracle实验。
'''
with (root/'README.md').open('x',encoding='utf-8') as f:f.write(text)
