from utils import MixIdentification
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import palettable
from statsmodels.api import OLS

warnings.filterwarnings('ignore')

# plotting params
alpha_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
font_prop_xlabels = {'family': 'Times New Roman', 'size': 25}
font_prop_ylabels = {'family': 'Times New Roman', 'size': 30}
font_prop_title = {'family': 'Times New Roman', 'size': 30}

data = pd.read_csv('data/data_comp.csv')
data.drop(columns='Unnamed: 0', inplace=True)
data['identifier'] = data.year * 100 + data.month
data['IndexVol_avg'] = data.OVX_avg / data.VIX_avg
data['IndexVol_end'] = data.OVX_end / data.VIX_end

####################
# identification-1 #
####################
var_names = ['OilVol', 'StockVol', 'VolRatio', 'WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory']
model = MixIdentification(
    y=np.array(data[var_names]),
    var_names=var_names,
    shock_names=['OilPriceUncertainty'],
    which='irf',
    which_var='OilVol',
    how='sum',
    reg_var='VolRatio',
    delta=10000,
    sign=1,
    reg_how='sum',
    period=1,  # 3
    pso=0,
    data_frequency='Monthly')
model.fit(lag=6)
rotation = model.identify()
model.point_estimate(h=40, with_vd=True)
model.boot_confid_intvl(h=40, n_path=100, irf_sig=[90, 95, 99], seed=3906)

# recover shocks
# epsilon = np.dot(np.linalg.inv(np.dot(A, model.rotation)), res.T)

plot_var = ['WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory']
model.plot_irf(var_list=plot_var, shock_list=['OilPriceUncertainty'], sigs=[90, 95], with_ci=True)
model.plot_vd(var_list=plot_var, shock_list=['OilPriceUncertainty'])

# plot IRF
h = 40
n_rows = 2
n_cols = 2
x_ticks = range(h + 1)
plt.figure(figsize=(18, 12), constrained_layout=True)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
vars = [3, 4, 5, 6]
sigs = [90, 99]
names = ['World oil production', 'World industrial production', 'Real oil price', 'World oil inventory']
for i in range(len(names)):
    color = palettable.tableau.BlueRed_6.mpl_colors[0]
    k = i + 1
    if i == 6:
        k = 8
    ax = plt.subplot(n_rows, n_cols, k)
    plt.plot(x_ticks, model.irf_point_estimate[vars[i], :] * 100, color=color, linewidth=3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    for sig, alpha in zip(sigs, alpha_list[1:]):
        plt.fill_between(x_ticks,
                         model.irf_confid_intvl[sig]['lower'][vars[i], :] * 100,
                         model.irf_confid_intvl[sig]['upper'][vars[i], :] * 100,
                         alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)
    plt.xlim(0, h)
    plt.xticks(list(range(0, h + 1, 5)))
    plt.title(names[i], font_prop_title, pad=5.)
    plt.tick_params(labelsize=22)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Palatino') for label in labels]
    ax.set_xlabel('Months', fontdict=font_prop_xlabels, labelpad=1.)
    ax.set_ylabel('%', fontdict=font_prop_xlabels, labelpad=0.5)
    plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
plt.savefig('decomp_irf.png', dpi=600)  # see line 252
plt.show()

# plot VD
h = 40
n_rows = 2
n_cols = 2
x_ticks = range(h + 1)
plt.figure(figsize=(18, 14), constrained_layout=True)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
vars = [3, 4, 5, 6]
sigs = [90, 99]
names = ['World oil production', 'World industrial production', 'Real oil price', 'World oil inventory']
for i in range(len(names)):
    color = palettable.tableau.BlueRed_6.mpl_colors[0]
    k = i + 1
    if i == 6:
        k = 8
    ax = plt.subplot(n_rows, n_cols, k)
    plt.plot(x_ticks, model.vd_point_estimate[vars[i], :] * 100, color=color, linewidth=3)
    plt.plot(x_ticks, (1 - model.vd_point_estimate[vars[i], :]) * 100, color='k', linewidth=3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    plt.xlim(0, h)
    plt.xticks(list(range(0, h + 1, 5)))
    plt.title(names[i], font_prop_title, pad=5.)
    plt.tick_params(labelsize=22)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Palatino') for label in labels]
    ax.set_xlabel('Months', fontdict=font_prop_xlabels, labelpad=1.)
    ax.set_ylabel('%', fontdict=font_prop_xlabels, labelpad=0.5)
    plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
plt.savefig('baseline_vd3.png', dpi=600)  # see line 252
plt.show()

####################
# identification-2 #
####################
var_names = ['OilVol', 'OVX_avg', 'VIX_avg', 'IndexVol_avg', 'WorldOilProd', 'IndustrialProd', 'WTISpotPrice',
             'OilInventory']
model = MixIdentification(
    y=np.array(data[var_names]),
    var_names=var_names,
    shock_names=['OilPriceUncertainty'],
    which='irf',
    which_var='OVX_avg',
    how='sum',
    reg_var='IndexVol_avg',
    delta=10000,
    sign=1,
    reg_how='sum',
    period=0,
    pso=0,
    data_frequency='Monthly')
model.fit(lag=6)
rotation = model.identify()
model.point_estimate(h=40, with_vd=True)
model.boot_confid_intvl(h=40, n_path=100, irf_sig=[90, 95, 99], seed=3906)
plot_var = ['WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory']
model.plot_irf(var_list=plot_var, shock_list=['OilPriceUncertainty'], sigs=[90, 99], with_ci=True)
model.plot_vd(var_list=plot_var, shock_list=['OilPriceUncertainty'])

# check aggregation
data.identifier = data.identifier.apply(lambda x: str(x))
xx = range(data.shape[0])
plt.figure(figsize=(10, 5), constrained_layout=True)
ax = plt.subplot(1, 2, 1)
plt.plot(xx, data.OVX_end, color=palettable.tableau.BlueRed_6.mpl_colors[0])
plt.plot(xx, data.OVX_avg, color=palettable.tableau.BlueRed_6.mpl_colors[1])
plt.tick_params(labelsize=12)
plt.xticks(list(xx)[::30], list(data.identifier)[::30], rotation=30)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Palatino') for label in labels]
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
plt.title('OVX', {'family': 'Palatino', 'size': 15, 'weight': 'normal'}, pad=2)
plt.legend(['end-of-month', 'average'], prop={'family': 'Palatino', 'size': 10}, loc=2)
ax = plt.subplot(1, 2, 2)
plt.plot(xx, data.VIX_end, color=palettable.tableau.BlueRed_6.mpl_colors[0])
plt.plot(xx, data.VIX_avg, color=palettable.tableau.BlueRed_6.mpl_colors[1])
plt.tick_params(labelsize=12)
plt.xticks(list(xx)[::30], list(data.identifier)[::30], rotation=30)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Palatino') for label in labels]
plt.title('VIX', {'family': 'Palatino', 'size': 15, 'weight': 'normal'}, pad=3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
plt.legend(['end-of-month', 'average'], prop={'family': 'Palatino', 'size': 10}, loc=0)
plt.savefig('comove.png', dpi=600)
plt.show()

# correlation
ovxe = np.array(data.OVX_end)
ovxa = np.array(data.OVX_avg)
vixe = np.array(data.VIX_end)
vixa = np.array(data.VIX_avg)
ovol = np.array(data.OilVol)
svol = np.array(data.StockVol)

np.corrcoef(ovol, ovxe)
np.corrcoef(ovol[1:], ovxe[:-1])
np.corrcoef(ovol[2:], ovxe[:-2])

np.corrcoef(ovol, ovxa)
np.corrcoef(ovol[1:], ovxa[:-1])
np.corrcoef(ovol[2:], ovxa[:-2])

np.corrcoef(svol, vixa)
np.corrcoef(svol[1:], vixa[:-1])
np.corrcoef(svol[2:], vixa[:-2])

np.corrcoef(svol, vixe)
np.corrcoef(svol[1:], vixe[:-1])
np.corrcoef(svol[2:], vixe[:-2])

np.corrcoef(vixa, vixe)
np.corrcoef(ovxa, ovxe)

np.corrcoef(svol[1:], vixe[:-1])
np.corrcoef(svol[2:], vixe[:-2])

h = 40
n_rows = 2
n_cols = 2
x_ticks = range(h + 1)
plt.figure(figsize=(18, 12), constrained_layout=True)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
vars = [4, 5, 6, 7]
sigs = [90, 99]
names = ['World oil production', 'World industrial production', 'Real oil price', 'World oil inventory']
for i in range(len(names)):
    color = palettable.tableau.BlueRed_6.mpl_colors[0]
    k = i + 1
    if i == 6:
        k = 8
    ax = plt.subplot(n_rows, n_cols, k)
    plt.plot(x_ticks, model.irf_point_estimate[vars[i], :] * 100, color=color, linewidth=3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    for sig, alpha in zip(sigs, alpha_list[1:]):
        plt.fill_between(x_ticks,
                         model.irf_confid_intvl[sig]['lower'][vars[i], :] * 100,
                         model.irf_confid_intvl[sig]['upper'][vars[i], :] * 100,
                         alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)
    plt.xlim(0, h)
    plt.xticks(list(range(0, h + 1, 5)))
    plt.title(names[i], font_prop_title, pad=5.)
    plt.tick_params(labelsize=22)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Palatino') for label in labels]
    ax.set_xlabel('Months', fontdict=font_prop_xlabels, labelpad=1.)
    ax.set_ylabel('%', fontdict=font_prop_xlabels, labelpad=0.5)
    plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
plt.savefig('ribust_irf6.png', dpi=600)  # see line 252
plt.show()

# decompose
ddata = data.copy()
ddata['lovx'] = ddata.OVX_avg.shift(1)
ddata['lvol'] = ddata.OilVol.shift(1)
ddata['lovx2'] = ddata['lovx'] ** 2
ddata.dropna(inplace=True)

decomp = OLS(endog=ddata['OilVol'], exog=ddata[['lovx2', 'lvol']], hasconst=True).fit()
ddata['uncer'] = decomp.fittedvalues
ddata['risk'] = ddata['lovx2'] - decomp.fittedvalues

ddata['xxx'] = ddata.uncer / ddata.VIX_avg
var_names = ['OilVol', 'uncer', 'VIX_avg', 'xxx', 'WorldOilProd', 'IndustrialProd', 'WTISpotPrice',
             'OilInventory']
model = MixIdentification(
    y=np.array(ddata[var_names]),
    var_names=var_names,
    shock_names=['OilPriceUncertainty'],
    which='irf',
    which_var='uncer',
    how='sum',
    reg_var='xxx',
    delta=10000,
    sign=1,
    reg_how='sum',
    period=0,
    pso=0,
    data_frequency='Monthly')
model.fit(lag=6)
rotation = model.identify()
model.point_estimate(h=40, with_vd=True)
model.boot_confid_intvl(h=40, n_path=100, irf_sig=[90, 95, 99], seed=3906)
plot_var = ['WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory']
model.plot_irf(var_list=plot_var, shock_list=['OilPriceUncertainty'], sigs=[90, 99], with_ci=True)
model.plot_vd(var_list=plot_var, shock_list=['OilPriceUncertainty'])

####################
# identification-3 #
####################

var_names = ['OilVol', 'StockVol', 'VolRatio', 'WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory',
             'OVX_avg', 'VIX_avg']
model = MixIdentification(
    y=np.array(data[var_names]),
    var_names=var_names,
    shock_names=['OilPriceUncertainty'],
    which='irf',
    which_var='OilVol',
    how='sum',
    reg_var='VolRatio',
    delta=10000,
    sign=1,
    reg_how='sum',
    period=6,
    pso=0,
    data_frequency='Monthly')
model.fit(lag=6)
rotation = model.identify()
model.point_estimate(h=40, with_vd=True)
model.boot_confid_intvl(h=40, n_path=100, irf_sig=[90, 95, 99], seed=3906)

plot_var = ['WorldOilProd', 'IndustrialProd', 'WTISpotPrice', 'OilInventory']
model.plot_irf(var_list=plot_var, shock_list=['OilPriceUncertainty'], sigs=[90, 95], with_ci=True)
model.plot_vd(var_list=plot_var, shock_list=['OilPriceUncertainty'])

h = 40
n_rows = 2
n_cols = 2
x_ticks = range(h + 1)
plt.figure(figsize=(18, 12), constrained_layout=True)
plt.subplots_adjust(wspace=0.3, hspace=0.4)
vars = [3, 4, 5, 6]
sigs = [90, 99]
names = ['World oil production', 'World industrial production', 'Real oil price', 'World oil inventory']
for i in range(len(names)):
    color = palettable.tableau.BlueRed_6.mpl_colors[0]
    k = i + 1
    if i == 6:
        k = 8
    ax = plt.subplot(n_rows, n_cols, k)
    plt.plot(x_ticks, model.irf_point_estimate[vars[i], :] * 100, color=color, linewidth=3)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    for sig, alpha in zip(sigs, alpha_list[1:]):
        plt.fill_between(x_ticks,
                         model.irf_confid_intvl[sig]['lower'][vars[i], :] * 100,
                         model.irf_confid_intvl[sig]['upper'][vars[i], :] * 100,
                         alpha=alpha, edgecolor=color, facecolor=color, linewidth=0)
    plt.xlim(0, h)
    plt.xticks(list(range(0, h + 1, 5)))
    plt.title(names[i], font_prop_title, pad=5.)
    plt.tick_params(labelsize=22)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Palatino') for label in labels]
    ax.set_xlabel('Months', fontdict=font_prop_xlabels, labelpad=1.)
    ax.set_ylabel('%', fontdict=font_prop_xlabels, labelpad=0.5)
    plt.grid(linestyle='--', linewidth=1.5, color='black', alpha=0.35)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
plt.savefig('augmented_irf.png', dpi=600)  # see line 252
plt.show()
