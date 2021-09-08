import FundamentalAnalysis as fa
import csv
import pandas as pd
import matplotlib.pyplot as plt

API_KEY = '50542cc27df9af35c4ed56bf358a68d1'

class DataProcess(object):
    def __init__(self) -> None:
        super().__init__()
        self.ticker = "AAPL"
        self.full_data = self.build_full_data()
    
    def get_pricing_data(self):
        stock_data = fa.stock_data_detailed(self.ticker, API_KEY)
        price_data = pd.DataFrame({'date': pd.to_datetime(stock_data.index.values), 
        'adj_close': stock_data['adjClose'].values, 'volume': stock_data['volume'].values})
        price_data = price_data.set_index('date')
        price_data = price_data.reindex(index=price_data.index[::-1])

        return price_data
    
    def get_ratio_data(self):
        ratios = fa.financial_ratios(self.ticker, API_KEY, period="quarter")
        financial_statements = fa.income_statement(self.ticker, API_KEY, period='quarter')
        ratio_data = pd.DataFrame({'DE Ratio': ratios.loc['debtEquityRatio'], 
        'Return on Equity': ratios.loc['returnOnEquity'], 'Price/Book': ratios.loc['priceToBookRatio'],
        'Gross Profit Margin': ratios.loc['grossProfitMargin'], 'Diluted EPS': financial_statements.loc['epsdiluted']})
        ratio_data['date'] = pd.to_datetime(ratio_data.index)
        ratio_data = ratio_data.set_index('date')

        return ratio_data
    
    def build_full_data(self):
        price_data = self.get_pricing_data()
        ratio_data = self.get_ratio_data()
        ratio_data_2 = ratio_data.reindex(price_data.index)

        full_data = pd.concat([price_data, ratio_data_2], axis=1)
        full_data = full_data.ffill()

        index = len(full_data[full_data.isna().any(axis=1)])

        return full_data[index:]