"""Generate editable SVG diagrams of the checked-in implementations.

No ML dependencies or external services. Text and geometry remain editable.
"""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/figures'
INK, MUTED, BLUE, AMBER, GREEN = '#172b4d', '#52637a', '#2463a6', '#986018', '#23775c'
COLORS = {'main': (BLUE, '#edf4fc'), 'aux': (AMBER, '#fff5e5'), 'target': (GREEN, '#edf7f2'), 'plain': ('#8593a5', '#f6f8fb')}


class Figure:
    def __init__(self, title, subtitle, height=840):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height}" viewBox="0 0 1400 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>',
                      '<defs>' + ''.join(f'<marker id="arrow-{name}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="{color}"/></marker>' for name, color in [('main', BLUE), ('aux', AMBER), ('target', GREEN), ('plain', MUTED)]) + '</defs>',
                      '<style>text{font-family:"Segoe UI","Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;fill:#172b4d} .small{fill:#52637a}</style>',
                      f'<rect width="1400" height="{height}" fill="white"/>']
        self.text(36, 43, title, 27, weight=600)
        self.text(36, 78, subtitle, 16, color=MUTED)
        self.parts.append('<path d="M36 98 H1364" stroke="#d8e1eb"/>')

    def text(self, x, y, value, size=17, anchor='start', color=INK, weight=400):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" style="fill:{color};font-weight:{weight}">{escape(value)}</text>')

    def box(self, x, y, w, h, lines, kind='main', size=17):
        color, fill = COLORS[kind]
        dash = ' stroke-dasharray="7 5"' if kind == 'aux' else ''
        self.parts.append('<g data-node-group="true">')
        self.parts.append(f'<rect data-node="true" x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{color}" stroke-width="1.6"{dash}/>')
        line_height = size + 10
        first = y + h / 2 - (len(lines) - 1) * line_height / 2 + size * .34
        for i, line in enumerate(lines):
            self.text(x + w / 2, first + i * line_height, line, size, anchor='middle', weight=600 if i == 0 else 400)
        self.parts.append('</g>')

    def arrow(self, points, kind='main', dashed=False):
        color = {'main': BLUE, 'aux': AMBER, 'target': GREEN, 'plain': MUTED}[kind]
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        vertices = ' '.join(f'{x},{y}' for x, y in points)
        self.parts.append(f'<polyline points="{vertices}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" marker-end="url(#arrow-{kind})"{dash}/>')

    def footer(self, lines):
        start = self.height - 35 - 25 * (len(lines) - 1)
        for i, line in enumerate(lines):
            self.text(36, start + i * 25, line, 15, color=MUTED)

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / name).open('w', encoding='utf-8', newline='\n') as handle:
            handle.write('\n'.join(self.parts + ['</svg>']) + '\n')


def einv2(causal):
    prefix = 'C' if causal else 'E'
    title = 'EINV2 · causal' if causal else 'EINV2 · offline / noncausal'
    f = Figure(title, f'{prefix}0 baseline；{prefix}1 + velocity；{prefix}2 + latent；{prefix}3 两项联合。图示 4s 输入，B 为 batch size。', 880)
    f.box(36, 300, 178, 125, ['FOA waveform', 'B × 4 × 96000', '24 kHz / 4 s'], size=17)
    f.box(256, 300, 252, 125, ['Feature + scalar', '4 log-mel + 3 IV', 'B × 7 × 161 × 256', 'hop = 25 ms'], size=16)
    f.arrow([(214, 363), (256, 363)])
    # The two backbones are shown as collapsed stages, with internal exchanges explicitly labeled.
    f.box(552, 130, 264, 154, ['SED CNN · 前 4 channels', 'DoubleConv × 4 stages', '64 → 128 → 256 → 512', 'pool (t,f): 2×2, 2×2,', '1×2, 1×2'], size=16)
    f.box(552, 445, 264, 154, ['DOA CNN · 全 7 channels', 'DoubleConv × 4 stages', '64 → 128 → 256 → 512', 'pool (t,f): 2×2, 2×2,', '1×2, 1×2'], size=16)
    f.arrow([(508, 338), (529, 338), (529, 207), (552, 207)])
    f.arrow([(508, 387), (529, 387), (529, 522), (552, 522)])
    f.box(566, 319, 238, 89, ['内部 3 次 soft-stitch', '发生于 stages 1–3 后', '顺序式 SED → DOA 更新'], kind='plain', size=15)
    f.arrow([(642, 284), (642, 319)])
    f.arrow([(722, 319), (722, 284)])
    f.arrow([(642, 408), (642, 445)])
    f.arrow([(722, 445), (722, 408)])
    for y in (145, 460):
        f.box(862, y, 233, 124, ['Freq mean → Transformer', '2 个独立 track encoders', '每个: 2 layers, d=512', '8 heads, FFN=1024'], size=15)
        f.arrow([(816, y + 62), (862, y + 62)])
    f.box(1141, 153, 223, 108, ['SED heads × 2', 'Linear 512 → 14 logits', 'B × 40 × 2 × 14', '推理时 sigmoid'], size=15)
    f.box(1141, 468, 223, 108, ['DOA heads × 2', 'Linear 512 → 3 + tanh', 'B × 40 × 2 × 3'], size=15)
    f.arrow([(1095, 207), (1141, 207)])
    f.arrow([(1095, 522), (1141, 522)])
    f.box(633, 655, 300, 91, [f'Velocity heads · {prefix}1 / {prefix}3', '每个 DOA track: Linear 512 → 3', '训练监督；不参与 SELD 解码'], kind='aux', size=15)
    f.box(1017, 655, 347, 91, [f'Latent branch · {prefix}2 / {prefix}3', '每 track: Linear 512 → 128 + LN', 'horizon-conditioned MLP → 128'], kind='aux', size=15)
    f.arrow([(965, 584), (965, 622), (783, 622), (783, 655)], 'aux', True)
    f.arrow([(965, 584), (965, 622), (1190, 622), (1190, 655)], 'aux', True)
    f.text(883, 306, 'output: B × 40 × 512', 14, color=MUTED)
    if causal:
        f.text(38, 480, 'Frontend: center=False + 左填充', 16)
        f.text(38, 509, 'Conv: 仅 past padding；channel LayerNorm', 16)
        f.text(38, 538, 'Attention: 上三角 mask；屏蔽未来', 16)
        f.text(38, 567, 'DOA head: 小初始化（weight × 0.1）', 16)
    else:
        f.text(38, 480, 'Frontend: centered STFT；legacy IV/scalar', 16)
        f.text(38, 509, 'Conv: 对称 padding；BatchNorm2d', 16)
        f.text(38, 538, 'Attention: 无 mask；全片段上下文', 16)
        f.text(38, 567, 'Latent loss 是 consistency，不称 causal prediction', 15)
    f.footer(['蓝色：主预测路径；橙色虚线：可选辅助分支。训练主损失为 0.5 BCEWithLogits + 0.5 DOA MSE，使用 frame-wise tPIT。',
              '辅助分支可能仍由当前 forward 计算，但不参与 SELD 输出解码；EMA teacher 仅用于训练/validation loss。',
              '示意图不宣称零时延：pooling、音频时间戳及 chunk 边界仍需端到端检验；soft-stitch 不画成同时对称交换。'])
    f.save('einv2_causal.svg' if causal else 'einv2_offline.svg')


def multi(causal):
    prefix = 'C' if causal else 'A'
    title = 'Multi-ACCDOA · causal' if causal else 'Multi-ACCDOA · noncausal'
    f = Figure(title, f'{prefix}0 baseline；{prefix}1 + velocity；{prefix}2 + latent；{prefix}3 联合。各变体共用 SeldModel，通过 flags 启用辅助 loss。', 810)
    f.box(36, 165, 166, 145, ['FOA features', 'B × 7 × 250 × 64', '4 log-mel + 3 IV', '5 s / hop 20 ms'], size=15)
    f.box(240, 142, 214, 191, ['CNN × 3 stages', 'past-only Conv3×3' if causal else 'symmetric Conv3×3', 'BatchNorm2d + ReLU', 'MaxPool + Dropout2d', 'channels: 64 / 64 / 64', 't-pool: 5 / 1 / 1', 'f-pool: 4 / 4 / 2'], size=15)
    f.box(492, 185, 156, 105, ['Reshape', 'B × 64 × 50 × 2', '→ B × 50 × 128'], size=15)
    gru_lines = ['GRU × 2 layers', 'unidirectional', 'hidden = 128', 'tanh → 128'] if causal else ['BiGRU × 2 layers', '128 per direction', 'tanh → [fwd,bwd]', 'fwd ⊙ bwd → 128']
    f.box(686, 165, 184, 145, gru_lines, size=15)
    f.box(908, 165, 192, 145, ['MHSA × 2 layers', 'd=128, heads=8', 'residual + LayerNorm', 'causal mask' if causal else 'no attention mask'], size=15)
    f.box(1138, 165, 226, 145, ['ACCDOA head', 'Linear 128 → 128 → 126', '最后 tanh', '无隐藏层激活'], size=15)
    for a, b in ((202, 240), (454, 492), (648, 686), (870, 908), (1100, 1138)):
        f.arrow([(a, 237), (b, 237)])
    f.box(1119, 365, 245, 129, ['Multi-ACCDOA output', 'B × 50 × 126', '= 3 tracks × xyz × 14 classes', '向量范数: activity；方向: DOA'], size=14)
    f.arrow([(1251, 310), (1251, 365)])
    f.box(502, 451, 270, 120, [f'Velocity · {prefix}1 / {prefix}3', 'Linear 128 → 126', 'B × 50 × 3 × 14 × 3', 'masked Smooth L1'], kind='aux', size=15)
    f.box(812, 451, 270, 120, [f'Latent · {prefix}2 / {prefix}3', 'Linear → 3 × 14 × 128', 'L2 normalize + predictor', 'MLP 129 → 256 → 128'], kind='aux', size=15)
    f.arrow([(1004, 310), (1004, 390), (637, 390), (637, 451)], 'aux', True)
    f.arrow([(1004, 390), (947, 390), (947, 451)], 'aux', True)
    f.text(675, 373, 'shared representation h: B × 50 × 128', 15, color=MUTED)
    f.box(36, 429, 405, 148, ['主监督: ADPIT MSE', '13 个候选 target；逐 frame / class 选最小', '3 output tracks ≠ 3 个固定声源身份', '辅助 mask 来自同一 ADPIT assignment'], kind='plain', size=15)
    f.text(38, 640, '当前 paper A/C configs 均使用 strict-causal frontend + train-fold-only normalization；A 模型仍是非因果 backbone。', 16)
    f.text(38, 674, 'CNN dropout=0.05；GRU dropout=0.05；MHSA dropout=0.05。C 模型的 BatchNorm 训练统计跨时间聚合。', 16)
    f.footer(['蓝色：主预测路径；橙色虚线：可选辅助分支，SELD 解码不依赖 auxiliary outputs / teacher。',
              '两类辅助 head 总会实例化；disabled 时冻结并跳过对应输出。非因果 latent loss 是 consistency，不宣称无 look-ahead。',
              'Causal 描述推理结构，不等于严格逐帧训练因果性或已验证零时延。'])
    f.save('multi_accdoa_causal.svg' if causal else 'multi_accdoa_noncausal.svg')


def velocity():
    f = Figure('方法示意 · velocity supervision', '辅助 head 预测单位方向向量的变化率，不是以 m/s 为单位的真实空间速度；Δt = 0.1 s。', 790)
    f.text(36, 141, 'EINV2 · C1 / C3 / E1 / E3', 22, weight=600)
    f.box(36, 184, 272, 132, ['GT metadata + DOA labels', 'identity = (class, source ID)', '匹配到 2 个内部 track slots', 'p 为 xyz 单位方向'], kind='plain', size=16)
    f.box(354, 184, 298, 132, ['有效相邻帧', '同一内部 slot 的 identity 相同', 'vₜ = (pₜ − pₜ₋₁) / 0.1', '首帧 / 不连续处不计 loss'], kind='plain', size=16)
    f.box(698, 184, 279, 132, ['tPIT 对齐', '沿主 SED + DOA 的置换', '对 velocity target / mask', '同步交换 track slots'], kind='aux', size=16)
    f.box(1023, 184, 341, 132, ['masked Smooth L1', 'DOA track features → Linear → v̂ₜ', '仅 valid mask 上平均', 'λᵥ = 0.2'], kind='aux', size=16)
    for a, b in ((308,354),(652,698),(977,1023)):
        f.arrow([(a,250),(b,250)], 'aux', True)
    f.text(36, 399, 'Multi-ACCDOA · C1 / C3 / A1 / A3', 22, weight=600)
    f.box(36, 442, 272, 132, ['ADPIT 主 loss', '13 个候选逐 frame / class', '选择 aligned ACCDOA target', '以及 assignment ID'], kind='plain', size=16)
    f.box(354, 442, 298, 132, ['有效相邻帧', '两端 activity > 0.5', '两端 assignment ID 相同', '不是 source identity 检查'], kind='plain', size=16)
    f.box(698, 442, 279, 132, ['方向变化 target', 'vₜ = (p̃ₜ − p̃ₜ₋₁) / 0.1', 'p̃: ADPIT-aligned target', '维度: track × class × xyz'], kind='aux', size=16)
    f.box(1023, 442, 341, 132, ['masked Smooth L1', 'shared h → Linear → v̂ₜ', '仅 valid mask 上平均', '当前 λᵥ = 0.05'], kind='aux', size=16)
    for a, b in ((308,354),(652,698),(977,1023)):
        f.arrow([(a,508),(b,508)], 'aux', True)
    f.text(36, 644, 'EINV2: L = 0.5 L_SED + 0.5 L_DOA + λᵥ L_velocity + λⱼ L_latent', 20)
    f.text(36, 682, 'Multi-ACCDOA: L = L_ADPIT + λᵥ L_velocity + λⱼ L_latent', 20)
    f.footer(['本图只描述实际实现：未加入轨迹积分、预测位置外推或额外 inference smoothing。',
              '禁用的辅助项不参与总 loss；静态 source 的有效相邻帧也可产生零 velocity target，并非只监督动态 source。'])
    f.save('velocity_supervision.svg')


def latent():
    f = Figure('方法示意 · JEPA-style latent prediction / consistency', 'h ∈ {1, 3, 5} label frames = 100 / 300 / 500 ms；future latent 仅作为训练目标，不作为主预测输入。', 850)
    f.box(36, 203, 179, 111, ['同一 feature chunk', 'x', 'online / target 共用'], kind='plain', size=16)
    f.box(260, 145, 228, 110, ['Online backbone θ', '取时间 t 的表示', 'EINV2: DOA track features'], size=15)
    f.box(534, 145, 217, 110, ['Projection gθ', 'latent dimension = 128', '得到 zₜ'], size=16)
    f.box(797, 145, 246, 110, ['Predictor qθ(zₜ, 0.1h)', 'MLP 129 → 256 → 128', 'GELU hidden activation', 'EINV2: tPIT canonicalize'], kind='aux', size=15)
    f.box(1131, 276, 233, 121, ['Masked cosine loss', '1 − cos(ẑₜ₊ₕ, sg(z̄ₜ₊ₕ))', '按有效 mask / horizon', '汇总后加入主损失'], kind='aux', size=15)
    f.arrow([(215,238),(237,238),(237,200),(260,200)])
    f.arrow([(488,200),(534,200)])
    f.arrow([(751,200),(797,200)], 'aux', True)
    f.arrow([(1043,200),(1100,200),(1100,306),(1131,306)], 'aux', True)
    f.box(260, 397, 228, 110, ['EMA target backbone θ̄', '同 architecture / eval()', 'no_grad / stop gradient'], kind='target', size=15)
    f.box(534, 397, 217, 110, ['Target projection gθ̄', '按目标时间 t+h 取 latent', '仅用作 loss target'], kind='target', size=15)
    f.box(797, 397, 246, 110, ['Alignment + stop-grad', 'EINV2: canonical tPIT slots', 'Multi: 固定 track/class slots'], kind='target', size=15)
    f.arrow([(215,280),(237,280),(237,452),(260,452)], 'target')
    f.arrow([(488,452),(534,452)], 'target')
    f.arrow([(751,452),(797,452)], 'target')
    f.arrow([(1043,452),(1100,452),(1100,367),(1131,367)], 'target')
    f.arrow([(374,255),(374,397)], 'target', True)
    f.arrow([(642,255),(642,397)], 'target', True)
    f.text(394, 322, 'EMA m=0.996', 15, color=GREEN)
    f.text(394, 350, '不是反向传播', 15, color=GREEN)
    f.box(36, 580, 627, 115, ['EINV2 mask / projection', '同一 GT identity 在 [t,t+h] 全区间连续且保持内部 slot', 'online tPIT 置换用于 canonicalize online / teacher slots', 'projection = Linear + LayerNorm；每 track 独立'], kind='plain', size=15)
    f.box(697, 580, 667, 115, ['Multi-ACCDOA mask / projection', '仅检查两端 activity 与 ADPIT assignment ID 相同', '没有检查区间内每一帧，也没有 source identity 监督', 'projection = Linear + L2 normalization；每 track/class latent'], kind='plain', size=15)
    f.footer(['Causal backbone 对应 future prediction；offline / noncausal backbone 已见未来上下文，只称 temporal latent consistency。',
              '蓝色：online 表示；橙色：可学习 predictor / loss；绿色：EMA reference。完整 teacher 由 model deepcopy 构造。',
              'EMA teacher 和未来 target 不参与 SELD 解码；这里不是标准 JEPA 复现声明，也没有独立 trajectory decoder。'])
    f.save('latent_prediction.svg')


def main():
    einv2(True)
    einv2(False)
    multi(True)
    multi(False)
    velocity()
    latent()
    print('Generated 6 editable SVG figures in docs/figures')


if __name__ == '__main__':
    main()
