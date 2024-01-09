import yfinance as yf
import pandas as pd
import datetime
from utils import trans_to_numeric
import numpy as np

data = pd.read_excel('data/data.xls')
data['Date'] = data.Date.apply(lambda x: datetime.datetime(x.year, x.month, x.day))
snp = yf.download('^GSPC')
wti = pd.read_excel('data/wti.xlsx')

# stock vol
snp['price_lag'] = snp['Adj Close'].shift(1)
snp['daily_ret'] = (snp['Adj Close'] - snp['price_lag']) / snp['price_lag']
snp['daily_ret_sq'] = snp['daily_ret'] ** 2
snp['month'] = snp.index.month
snp['year'] = snp.index.year
stockvol = snp.groupby(['year', 'month']).agg({'daily_ret_sq': 'sum'})
stockvol['year'] = stockvol.index.get_level_values(0)
stockvol['month'] = stockvol.index.get_level_values(1)
stockvol.index = pd.Index(range(stockvol.shape[0]))
stockvol.rename(columns={'daily_ret_sq': 'StockVol'}, inplace=True)

# oil vol
wti['year'] = wti.Date.apply(lambda x: x.year)
wti['month'] = wti.Date.apply(lambda x: x.month)
wti = wti[['year', 'month', 'price']].copy()
wti = trans_to_numeric(wti)
wti['price_lag'] = wti['price'].shift(1)
wti['daily_ret'] = (wti.price - wti.price_lag) / wti.price_lag
wti['daily_ret_sq'] = wti['daily_ret'] ** 2
wti.dropna(subset='daily_ret_sq', inplace=True)
wtivol = wti.groupby(['year', 'month']).agg({'daily_ret_sq': 'sum'})
wtivol['year'] = wtivol.index.get_level_values(0)
wtivol['month'] = wtivol.index.get_level_values(1)
wtivol.index = pd.Index(range(wtivol.shape[0]))
wtivol.rename(columns={'daily_ret_sq': 'OilVol'}, inplace=True)

# merge data
data['year'] = data.Date.apply(lambda x: x.year)
data['month'] = data.Date.apply(lambda x: x.month)
data = pd.merge(data, stockvol, on=['year', 'month'])
data = pd.merge(data, wtivol, on=['year', 'month'])
data = data[['Date', 'year', 'month', 'OVX_avg', 'OVX_sum', 'OVX_end', 'VIX_avg', 'VIX_sum',
             'VIX_end', 'WorldOilProd', 'CrudeOilStock', 'PetroleumStockUS',
             'PetroleumStockOECD', 'IndustrialProd', 'EconActivityShock',
             'OilConsumptionDemandShock', 'InventoryDemandShock', 'OilSupplyShock',
             'WTISpotPrice', 'GPR', 'CPI', 'StockVol', 'OilVol']].copy()

# check some facts
np.corrcoef(data[['StockVol', 'OilVol']].T)
np.corrcoef(data[['StockVol', 'VIX_end']].T)
np.corrcoef(data[['StockVol', 'VIX_avg']].T)
np.corrcoef(data[['StockVol', 'VIX_sum']].T)
np.corrcoef(data[['OilVol', 'OVX_end']].T)
np.corrcoef(data[['OilVol', 'OVX_avg']].T)
np.corrcoef(data[['OilVol', 'OVX_sum']].T)

# some transformation
data['VolRatio'] = data.OilVol / data.StockVol
data.WTISpotPrice = np.log((data.WTISpotPrice / data.CPI) * 100)
data.WorldOilProd = np.log(data.WorldOilProd)
data.IndustrialProd = np.log(data.IndustrialProd)
data['ScaleFactor'] = data.PetroleumStockOECD / data.PetroleumStockUS
data['OilInventory'] = data.CrudeOilStock * data.ScaleFactor
data.OilInventory = np.log(data.OilInventory)
data.to_csv('data/data_comp.csv')
