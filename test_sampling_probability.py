"""
采样放大机制 (Sample Amplification) 采样概率可视化

Section 3.2 公式（eps2 = 0 的特殊情况）：
    p = (e^{ε'} - 1) / [(e^{ε/2} - 1) * (e^{ε' - ε/2} + 1)]

本脚本：
  固定 eps2 = 0，对 eps = 16, 18, 20, 24，
  扫描不同 eps'，计算采样概率 p 并画图。
  
  输出：
    1. 数据表（控制台）
    2. 三张图：MLDP尺度、MLDP尺度放大过渡区、Pure-DP尺度
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ============================================================
# 核心公式
# ============================================================

def compute_p_section32(eps, eps_prime):
    """
    Section 3.2 公式（eps2 = 0 的特殊情况）：
    p = (e^{ε'} - 1) / [(e^{ε/2} - 1) * (e^{ε' - ε/2} + 1)]
    """
    numerator = np.exp(eps_prime) - 1
    denominator = (np.exp(eps / 2) - 1) * (np.exp(eps_prime - eps / 2) + 1)
    p = numerator / denominator
    return np.clip(p, 0.0, 1.0)


def compute_p_general(eps1, eps2, eps_prime):
    """
    Section 3.3 一般公式：
    p = (e^{eps'} - e^{eps2}) / [(e^{eps1} - e^{eps'}) * e^{-(eps1+eps2)/2} + (e^{eps'} - e^{eps2})]
    """
    e1 = np.exp(eps1)
    e2 = np.exp(eps2)
    ep = np.exp(eps_prime)
    numerator = ep - e2
    denominator = (e1 - ep) * np.exp(-(eps1 + eps2) / 2) + (ep - e2)
    p = numerator / denominator
    return np.clip(p, 0.0, 1.0)


# ============================================================
# 常量
# ============================================================
D_MAX = 2.892667  # BERT word embedding 最大欧式距离
EPS_MLDP_LIST = [16, 18, 20, 24]

# ============================================================
# 1. 计算数据（使用更密的采样点以获得平滑曲线）
# ============================================================
print("=" * 75)
print("采样放大机制：采样概率 p 随目标隐私预算 ε' 的变化")
print("=" * 75)
print(f"  d_max = {D_MAX}")
print(f"  eps2 (MLDP) = 0  (均匀随机)")
print(f"  eps  (MLDP) = {EPS_MLDP_LIST}")
print()

all_data = {}

for eps_mldp in EPS_MLDP_LIST:
    eps_pure = eps_mldp * D_MAX

    # 密采样：步长 0.1
    eps_prime_mldp_arr = np.arange(0.1, eps_mldp, 0.1)
    eps_prime_pure_arr = eps_prime_mldp_arr * D_MAX
    p_arr = compute_p_section32(eps_pure, eps_prime_pure_arr)

    all_data[eps_mldp] = {
        'eps_prime_mldp': eps_prime_mldp_arr,
        'eps_prime_pure': eps_prime_pure_arr,
        'p': p_arr,
    }

    # 打印关键数据点（步长 1）
    print(f"\n--- ε (MLDP) = {eps_mldp}  |  ε (pure-DP) = {eps_pure:.2f} ---")
    print(f"{'ε_MLDP':>8}  {'ε_pure':>10}  {'p':>12}")
    print("-" * 35)
    for em, ep, pv in zip(eps_prime_mldp_arr, eps_prime_pure_arr, p_arr):
        if abs(em - round(em)) < 0.05:  # 只打印整数点
            print(f"{em:>8.1f}  {ep:>10.2f}  {pv:>12.8f}")

# ============================================================
# 2. 画图
# ============================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['axes.unicode_minus'] = False

colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
markers = ['o', 's', '^', 'D']
linestyles = ['-', '--', '-.', ':']

# ==========================================
# 图1：MLDP 尺度全景图（论文主图）
# ==========================================
fig1, ax1 = plt.subplots(figsize=(9, 6))

for i, eps_mldp in enumerate(EPS_MLDP_LIST):
    data = all_data[eps_mldp]
    ax1.plot(data['eps_prime_mldp'], data['p'],
             color=colors[i], linestyle=linestyles[i], linewidth=2.5,
             label=r'$\varepsilon = %d$' % eps_mldp)

# 标注 ε' = ε/2 时 p = 0.5 的特殊点
for i, eps_mldp in enumerate(EPS_MLDP_LIST):
    half_eps = eps_mldp / 2
    ax1.plot(half_eps, 0.5, marker='*', markersize=16,
             color=colors[i], zorder=5,
             markeredgecolor='black', markeredgewidth=0.8)

ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax1.text(0.5, 0.53, r"$p = 0.5$  when  $\varepsilon' = \varepsilon/2$",
         fontsize=11, color='gray', style='italic')

ax1.set_xlabel(r"Target Privacy Budget  $\varepsilon'$  (MLDP)", fontsize=14)
ax1.set_ylabel(r"Sampling Probability  $p$", fontsize=14)
ax1.set_title(r"Sample Amplification: $p$ vs $\varepsilon'$", fontsize=15, fontweight='bold')
ax1.legend(fontsize=13, title=r'Original $\varepsilon$', title_fontsize=13,
           loc='center right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 25)
ax1.set_ylim(-0.02, 1.05)
ax1.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("sampling_probability_mldp.png", dpi=200, bbox_inches='tight')
plt.savefig("sampling_probability_mldp.pdf", dpi=200, bbox_inches='tight')
print("\n图1已保存：sampling_probability_mldp.png / .pdf")

# ==========================================
# 图2：MLDP 尺度 - 过渡区域放大图
# ==========================================
fig2, ax2 = plt.subplots(figsize=(9, 6))

for i, eps_mldp in enumerate(EPS_MLDP_LIST):
    data = all_data[eps_mldp]
    # 只画过渡区域：ε/2 ± 4
    half = eps_mldp / 2
    mask = (data['eps_prime_mldp'] >= half - 4) & (data['eps_prime_mldp'] <= half + 4)
    ax2.plot(data['eps_prime_mldp'][mask], data['p'][mask],
             color=colors[i], marker=markers[i], markersize=5,
             markevery=5,
             linestyle=linestyles[i], linewidth=2.5,
             label=r'$\varepsilon = %d$  ($\varepsilon/2 = %d$)' % (eps_mldp, eps_mldp // 2))

# 标注 p = 0.5 线
ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.2)
ax2.text(4.5, 0.53, r"$p = 0.5$", fontsize=12, color='gray')

# 标注特殊点
for i, eps_mldp in enumerate(EPS_MLDP_LIST):
    half_eps = eps_mldp / 2
    ax2.plot(half_eps, 0.5, marker='*', markersize=16,
             color=colors[i], zorder=5,
             markeredgecolor='black', markeredgewidth=0.8)

ax2.set_xlabel(r"Target Privacy Budget  $\varepsilon'$  (MLDP)", fontsize=14)
ax2.set_ylabel(r"Sampling Probability  $p$", fontsize=14)
ax2.set_title(r"Transition Region (zoomed around $\varepsilon' = \varepsilon/2$)", fontsize=15, fontweight='bold')
ax2.legend(fontsize=12, loc='center right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(4, 16)
ax2.set_ylim(-0.02, 1.05)
ax2.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("sampling_probability_transition.png", dpi=200, bbox_inches='tight')
plt.savefig("sampling_probability_transition.pdf", dpi=200, bbox_inches='tight')
print("图2已保存：sampling_probability_transition.png / .pdf")

# ==========================================
# 图3：Pure-DP 尺度
# ==========================================
fig3, ax3 = plt.subplots(figsize=(9, 6))

for i, eps_mldp in enumerate(EPS_MLDP_LIST):
    data = all_data[eps_mldp]
    ax3.plot(data['eps_prime_pure'], data['p'],
             color=colors[i], linestyle=linestyles[i], linewidth=2.5,
             label=r'$\varepsilon_{MLDP} = %d$  ($\varepsilon_{pure} = %.1f$)' % (eps_mldp, eps_mldp * D_MAX))

ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)

ax3.set_xlabel(r"Target Privacy Budget  $\varepsilon'$  (Pure-DP)", fontsize=14)
ax3.set_ylabel(r"Sampling Probability  $p$", fontsize=14)
ax3.set_title(r"Sample Amplification: $p$ vs $\varepsilon'$ (Pure-DP scale)", fontsize=15, fontweight='bold')
ax3.legend(fontsize=12, loc='center right')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.02, 1.05)
ax3.tick_params(labelsize=12)

plt.tight_layout()
plt.savefig("sampling_probability_pure_dp.png", dpi=200, bbox_inches='tight')
plt.savefig("sampling_probability_pure_dp.pdf", dpi=200, bbox_inches='tight')
print("图3已保存：sampling_probability_pure_dp.png / .pdf")

# ============================================================
# 3. 验证特殊情况
# ============================================================
print("\n" + "=" * 75)
print("验证特殊情况")
print("=" * 75)

for eps_mldp in EPS_MLDP_LIST:
    eps_pure = eps_mldp * D_MAX
    p_zero = compute_p_section32(eps_pure, 1e-10)
    p_half = compute_p_section32(eps_pure, eps_pure / 2)
    print(f"  ε={eps_mldp:>2d} (pure={eps_pure:>6.2f}):  "
          f"p(ε'→0) = {p_zero:.8f},  "
          f"p(ε'=ε/2) = {p_half:.8f}")

# ============================================================
# 4. 输出 CSV 数据（方便后续使用）
# ============================================================
csv_path = "sampling_probability_data.csv"
with open(csv_path, 'w') as f:
    f.write("eps_mldp,eps_prime_mldp,eps_prime_pure,p\n")
    for eps_mldp in EPS_MLDP_LIST:
        data = all_data[eps_mldp]
        for em, ep, pv in zip(data['eps_prime_mldp'], data['eps_prime_pure'], data['p']):
            f.write(f"{eps_mldp},{em:.1f},{ep:.2f},{pv:.10f}\n")
print(f"\nCSV 数据已保存：{csv_path}")

plt.show()
