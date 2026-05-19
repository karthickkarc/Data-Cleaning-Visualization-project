import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── 1. GENERATE RAW (MESSY) DATASET ───────────────────────────────────────────
n = 500
departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
raw = pd.DataFrame({
    'employee_id':   list(range(1001, 1001+n)),
    'department':    np.random.choice(departments, n),
    'age':           np.random.randint(22, 60, n).astype(float),
    'salary':        np.random.normal(75000, 20000, n),
    'years_exp':     np.random.randint(0, 30, n).astype(float),
    'performance':   np.random.choice(['Low','Medium','High','Excellent'], n,
                                       p=[0.15,0.40,0.30,0.15]),
    'satisfaction':  np.random.uniform(1, 10, n),
    'overtime_hrs':  np.random.exponential(5, n),
})

# Inject issues
idx_miss = np.random.choice(n, 60, replace=False)
raw.loc[idx_miss[:30], 'salary']      = np.nan
raw.loc[idx_miss[30:], 'years_exp']   = np.nan

idx_out = np.random.choice(n, 10, replace=False)
raw.loc[idx_out[:5],  'salary']       = np.random.uniform(200000, 400000, 5)
raw.loc[idx_out[5:],  'overtime_hrs'] = np.random.uniform(80, 120, 5)

idx_dup = np.random.choice(n, 20, replace=False)
raw = pd.concat([raw, raw.iloc[idx_dup]], ignore_index=True)

raw['age'] = raw['age'].apply(lambda x: -x if np.random.random() < 0.02 else x)
raw.to_csv('/home/claude/raw_employee_data.csv', index=False)

# ─── 2. CLEANING ────────────────────────────────────────────────────────────────
df = raw.copy()
before_rows = len(df)

# Duplicates
df.drop_duplicates(subset='employee_id', keep='first', inplace=True)
dup_removed = before_rows - len(df)

# Negative ages
neg_age = (df['age'] < 0).sum()
df['age'] = df['age'].abs()

# Missing values
miss_before = df[['salary','years_exp']].isna().sum()
df['salary'].fillna(df.groupby('department')['salary'].transform('median'), inplace=True)
df['years_exp'].fillna(df['years_exp'].median(), inplace=True)

# Outliers (IQR)
def remove_outliers_iqr(series, factor=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(Q1 - factor*IQR, Q3 + factor*IQR)

df['salary']      = remove_outliers_iqr(df['salary'])
df['overtime_hrs']= remove_outliers_iqr(df['overtime_hrs'])

after_rows = len(df)
df.to_csv('/home/claude/clean_employee_data.csv', index=False)

# ─── 3. COMPUTE STATS ────────────────────────────────────────────────────────────
dept_stats = df.groupby('department').agg(
    avg_salary   = ('salary',       'mean'),
    avg_exp      = ('years_exp',    'mean'),
    avg_sat      = ('satisfaction', 'mean'),
    avg_ot       = ('overtime_hrs', 'mean'),
    count        = ('employee_id',  'count'),
).round(1).reset_index()

perf_dist = df['performance'].value_counts(normalize=True).mul(100).round(1)
sat_by_dept = df.groupby('department')['satisfaction'].mean().sort_values()

# ─── 4. FIGURE ──────────────────────────────────────────────────────────────────
BG      = '#0f1117'
CARD    = '#1a1d27'
ACCENT1 = '#6c63ff'
ACCENT2 = '#ff6b6b'
ACCENT3 = '#43e97b'
ACCENT4 = '#f7971e'
ACCENT5 = '#38f9d7'
TEXT    = '#e8eaf0'
MUTED   = '#6b7280'

DEPT_COLORS = {'Engineering': ACCENT1, 'Marketing': ACCENT2,
               'Sales': ACCENT3,       'HR': ACCENT4, 'Finance': ACCENT5}
PERF_COLORS = {'Low': ACCENT2, 'Medium': '#f7971e',
               'High': ACCENT1, 'Excellent': ACCENT3}

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'text.color': TEXT, 'axes.labelcolor': TEXT,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.edgecolor': '#2d3148', 'grid.color': '#2d3148',
    'grid.linewidth': 0.6, 'font.family': 'DejaVu Sans',
})

fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = fig.add_gridspec(4, 4, hspace=0.52, wspace=0.38,
                      left=0.06, right=0.97, top=0.91, bottom=0.05)

# ── Title banner
ax_title = fig.add_axes([0, 0.93, 1, 0.07])
ax_title.set_facecolor(BG); ax_title.axis('off')
ax_title.text(0.5, 0.70, 'EMPLOYEE ANALYTICS DASHBOARD',
              ha='center', va='center', fontsize=26, fontweight='bold',
              color=TEXT, transform=ax_title.transAxes)
ax_title.text(0.5, 0.18,
              f'Cleaned dataset  •  {after_rows} records  •  {len(departments)} departments  •  Quality issues resolved: {dup_removed} duplicates • {neg_age} invalid ages • {miss_before.sum()} missing values • outliers capped',
              ha='center', va='center', fontsize=9, color=MUTED,
              transform=ax_title.transAxes)

# ── KPI cards (row 0, spans)
kpi_data = [
    ('EMPLOYEES',    f'{after_rows}',          ACCENT1, '👥'),
    ('AVG SALARY',   f'${df.salary.mean()/1e3:.1f}K', ACCENT3, '💰'),
    ('AVG TENURE',   f'{df.years_exp.mean():.1f} yrs', ACCENT4, '📅'),
    ('AVG SATISFACTION', f'{df.satisfaction.mean():.2f}/10', ACCENT2, '⭐'),
]
for i,(label, val, col, ico) in enumerate(kpi_data):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(CARD); ax.axis('off')
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.add_patch(mpatches.FancyBboxPatch((0,0),1,1, boxstyle='round,pad=0.02',
                 facecolor=CARD, edgecolor=col, linewidth=1.5,
                 transform=ax.transAxes, clip_on=False, zorder=0))
    ax.text(0.12, 0.62, ico, fontsize=22, transform=ax.transAxes, va='center')
    ax.text(0.5,  0.62, val, fontsize=20, fontweight='bold', color=col,
            transform=ax.transAxes, ha='center', va='center')
    ax.text(0.5,  0.22, label, fontsize=8, color=MUTED,
            transform=ax.transAxes, ha='center', va='center')

# ── 1. Salary by Department (horizontal bar) — row1, col0-1
ax1 = fig.add_subplot(gs[1, :2])
bars = ax1.barh(dept_stats['department'],
                dept_stats['avg_salary']/1000,
                color=[DEPT_COLORS[d] for d in dept_stats['department']],
                height=0.55, edgecolor='none')
for bar, val in zip(bars, dept_stats['avg_salary']):
    ax1.text(bar.get_width()+0.4, bar.get_y()+bar.get_height()/2,
             f'${val/1e3:.1f}K', va='center', fontsize=9, color=TEXT)
ax1.set_xlabel('Average Salary (K)', fontsize=9)
ax1.set_title('Avg Salary by Department', fontsize=11, fontweight='bold', pad=10, color=TEXT)
ax1.grid(axis='x', alpha=0.4); ax1.set_axisbelow(True)
ax1.spines[['top','right']].set_visible(False)

# ── 2. Performance Distribution (donut) — row1, col2-3
ax2 = fig.add_subplot(gs[1, 2:])
order = ['Low','Medium','High','Excellent']
sizes = [perf_dist.get(k,0) for k in order]
colors = [PERF_COLORS[k] for k in order]
wedges, texts, autos = ax2.pie(
    sizes, labels=order, colors=colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.5, edgecolor=BG, linewidth=2),
    textprops={'color': TEXT, 'fontsize': 9})
ax2.set_title('Performance Distribution', fontsize=11, fontweight='bold', pad=10, color=TEXT)

# ── 3. Salary Distribution (violin) — row2, col0-1
ax3 = fig.add_subplot(gs[2, :2])
dept_order = dept_stats.sort_values('avg_salary')['department'].tolist()
data_by_dept = [df[df['department']==d]['salary'].values/1000 for d in dept_order]
vp = ax3.violinplot(data_by_dept, positions=range(len(dept_order)),
                    showmedians=True, showextrema=False)
for i,(body, d) in enumerate(zip(vp['bodies'], dept_order)):
    body.set_facecolor(DEPT_COLORS[d]); body.set_alpha(0.7)
vp['cmedians'].set_color(TEXT); vp['cmedians'].set_linewidth(2)
ax3.set_xticks(range(len(dept_order)))
ax3.set_xticklabels(dept_order, fontsize=8)
ax3.set_ylabel('Salary (K)', fontsize=9)
ax3.set_title('Salary Distribution by Department', fontsize=11, fontweight='bold', pad=10, color=TEXT)
ax3.grid(axis='y', alpha=0.4); ax3.set_axisbelow(True)
ax3.spines[['top','right']].set_visible(False)

# ── 4. Satisfaction by Dept (lollipop) — row2, col2-3
ax4 = fig.add_subplot(gs[2, 2:])
sat_sorted = sat_by_dept.sort_values()
colors4 = [DEPT_COLORS[d] for d in sat_sorted.index]
ax4.hlines(range(len(sat_sorted)), 0, sat_sorted.values, color=MUTED, linewidth=1.5, alpha=0.5)
ax4.scatter(sat_sorted.values, range(len(sat_sorted)), color=colors4, s=100, zorder=3)
ax4.set_yticks(range(len(sat_sorted)))
ax4.set_yticklabels(sat_sorted.index, fontsize=9)
ax4.set_xlabel('Avg Satisfaction Score', fontsize=9)
ax4.set_title('Satisfaction by Department', fontsize=11, fontweight='bold', pad=10, color=TEXT)
ax4.grid(axis='x', alpha=0.4); ax4.set_axisbelow(True)
ax4.spines[['top','right']].set_visible(False)
for val, pos in zip(sat_sorted.values, range(len(sat_sorted))):
    ax4.text(val+0.05, pos, f'{val:.2f}', va='center', fontsize=8, color=TEXT)

# ── 5. Experience vs Salary scatter — row3, col0-1
ax5 = fig.add_subplot(gs[3, :2])
sc_colors = [DEPT_COLORS[d] for d in df['department']]
ax5.scatter(df['years_exp'], df['salary']/1000,
            c=sc_colors, alpha=0.45, s=18, edgecolors='none')
m, b, r, *_ = stats.linregress(df['years_exp'], df['salary']/1000)
xs = np.linspace(df['years_exp'].min(), df['years_exp'].max(), 200)
ax5.plot(xs, m*xs+b, color=TEXT, linewidth=2, linestyle='--', alpha=0.8)
ax5.set_xlabel('Years of Experience', fontsize=9)
ax5.set_ylabel('Salary (K)', fontsize=9)
ax5.set_title(f'Experience vs Salary  (r={r:.2f})', fontsize=11, fontweight='bold', pad=10, color=TEXT)
ax5.grid(alpha=0.3); ax5.set_axisbelow(True)
ax5.spines[['top','right']].set_visible(False)
handles = [mpatches.Patch(color=DEPT_COLORS[d], label=d) for d in departments]
ax5.legend(handles=handles, fontsize=7, framealpha=0.2, loc='upper left',
           facecolor=CARD, edgecolor='none', labelcolor=TEXT)

# ── 6. Overtime Heatmap — row3, col2-3
ax6 = fig.add_subplot(gs[3, 2:])
perf_order = ['Low','Medium','High','Excellent']
heat_data = df.groupby(['department','performance'])['overtime_hrs'].mean().unstack(fill_value=0)
heat_data = heat_data.reindex(columns=perf_order, fill_value=0)
sns.heatmap(heat_data, ax=ax6, cmap='YlOrRd',
            annot=True, fmt='.1f', annot_kws={'size':8, 'color': '#111'},
            linewidths=0.5, linecolor=BG,
            cbar_kws={'shrink':0.7, 'pad':0.02})
ax6.set_title('Avg Overtime Hours\n(Dept × Performance)', fontsize=11,
              fontweight='bold', pad=10, color=TEXT)
ax6.set_xlabel('Performance', fontsize=9)
ax6.set_ylabel('Department', fontsize=9)
ax6.tick_params(axis='both', labelsize=8)

plt.savefig('/home/claude/dashboard.png', dpi=160, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.close()

print(f"✅ Raw rows: {before_rows} → Clean rows: {after_rows}")
print(f"   Duplicates removed : {dup_removed}")
print(f"   Negative ages fixed: {neg_age}")
print(f"   Missing values filled: {miss_before.sum()}")
print(f"   Outliers capped via IQR method")
print("✅ Dashboard saved → /home/claude/dashboard.png")
