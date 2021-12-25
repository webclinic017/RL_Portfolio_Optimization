from pyalgotrade import strategy
from pyalgotrade import barfeed
from pyalgotrade.barfeed import csvfeed
from pyalgotrade.bar import Frequency
# from pyalgotrade.feed import csvfeed


# def validate_forward(self, x, y_prev, y_gt):
#     valid_loss = 0
        
#     _, input_encoded = self.Encoder(Variable(x.type(torch.FloatTensor).to(self.device)))
        
#     d_n = self.Decoder._init_states(input_encoded)
#     c_n = self.Decoder._init_states(input_encoded)

#     y_prev = Variable(y_prev.type(torch.FloatTensor).to(self.device))

#     for t in range(self.T_predict):
#         if t == 0:
#             y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=True)

#         else:
#             y_pred, d_n, c_n = self.Decoder(input_encoded, y_prev, d_n, c_n, initial=False)
        
#         y_true = Variable(y_gt[:, t].type(torch.FloatTensor).to(self.device))

#         y_true = y_true.view(-1, 1)
#         valid_loss += self.criterion(y_pred, y_true)

#         y_prev = y_pred.detach()
    
#     return valid_loss.item()
class Predictor:
    def predict(self, date_time_series, close_data_series):
        pass


class PredictorStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument, predictor, tick_size=0.01, lot_size=100, max_position=10000):
        super(PredictorStrategy, self).__init__(feed)
        self.instrument = instrument
        self.last_position = None
        self.initialPortfolio = self.getBroker().getEquity()
        self.predictor = predictor
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.max_position = max_position

    def onStart(self):
        self.info(f"Starting run. Initial portfolio value: $%.f" % self.initialPortfolio)

    def onFinish(self, bars):
        broker = self.getBroker()
        portfolio = broker.getEquity()
        profit = portfolio - self.initialPortfolio
        returns = profit / self.initialPortfolio
        date_times = self.getFeed().getDataSeries().getDateTimes()
        trading_days = len(set([date_time.date() for date_time in date_times]))
        year_fraction = trading_days / 252.0
        daily_returns = (returns + 1.0) ** (1.0 / trading_days) - 1.0
        annual_returns = (returns + 1.0) ** (1.0 / year_fraction) - 1.0
        self.info(f"Completed run. Final portfolio value: $%.2f. Final PnL: $%.2f. Daily return: %.2f%%. Annual return: %.2f%%. Current position %.1f"
                  % (portfolio, profit, daily_returns * 100.0, annual_returns * 100.0, broker.getShares(self.instrument)))

    def onEnterOk(self, position):
        order = position.getEntryOrder()
        execInfo = order.getExecutionInfo()
        self.info(f"%s at $%.2f" % ("BUY" if order.isBuy() else "SELL", execInfo.getPrice()))

    def onExitOk(self, position):
        order = position.getExitOrder()
        execInfo = order.getExecutionInfo()
        self.info(f"%s at $%.2f" % ("BUY" if order.isBuy() else "SELL", execInfo.getPrice()))
        self.last_position = None

    def onBars(self, bars):
        if self.last_position is not None and self.last_position.entryActive():
            last_order = self.last_position.getEntryOrder()
            self.debug(f"CANCEL open order to %s at $%.2f" % ("BUY" if last_order.isBuy() else "SELL", last_order.getLimitPrice()))
            self.last_position.cancelEntry()
            self.last_position = None

        bar = bars[self.instrument]
        close_price = bar.getClose()
        self.debug(f"New price: $%.2f" % close_price)

        feed_series = self.getFeed().getDataSeries()
        prediction = self.predictor.predict(feed_series.getDateTimes(), feed_series.getCloseDataSeries())

        broker = self.getBroker()
        current_shares = broker.getShares(self.instrument)

        buy_target_price = close_price + self.tick_size / 2.0
        sell_target_price = close_price - self.tick_size / 2.0
        if prediction is None:
            return
        elif prediction > buy_target_price:
            cash = broker.getCash()
            buy_capacity = cash / buy_target_price
            max_shares_allowed = self.max_position - current_shares

            quantity = max(min(self.lot_size, buy_capacity, max_shares_allowed), 0)
            self.last_position = self.enterLongLimit(self.instrument, buy_target_price, quantity)
            self.debug(f"Submitting order to BUY %.1f at $%.2f" % (quantity, buy_target_price))
        elif prediction < sell_target_price:
            max_shares_allowed = self.max_position + current_shares

            quantity = min(self.lot_size, max_shares_allowed)
            self.last_position = self.enterShortLimit(self.instrument, sell_target_price, quantity)
            self.debug(f"Submitting order to SELL %.1f at $%.2f" % (quantity, sell_target_price))


class DummyPredictor(Predictor):

    def predict(self, date_time_series, close_data_series):
        if len(close_data_series) > 5:
            return close_data_series[-1] + (close_data_series[-1] - close_data_series[-2])


class LongShortStrategy(strategy.BacktestingStrategy):
    def __init__(self, feed, instrument):
        super(LongShortStrategy, self).__init__(feed)
        self.instrument = instrument
        # self.setUseAdjustedValues(True)
        self.position = None

    def onEnterOk(self, position):
        execInfo = position.getEntryOrder().getExecutionInfo()
        self.info(f"BUY at $%.2f" % (execInfo.getPrice()))

    def onExitOk(self, position):
        execInfo = position.getExitOrder().getExecutionInfo()
        self.info(f"SELL at $%.2f" % (execInfo.getPrice()))
        self.position = None

    def onBars(self, bars):
        bar = bars[self.instrument]
        self.info(bar.getClose())

        if len(self.getFeed().getDataSeries().getCloseDataSeries()) > 5: # wait for lookback
            print(self.getFeed().getDataSeries().getCloseDataSeries()[-3:])

        if self.position is None:
            close = bar.getClose()
            broker = self.getBroker()
            cash = broker.getCash()

            quantity = cash / close
            self.position = self.enterLong(self.instrument, quantity)


# feed = yahoofeed.Feed()
# feed.addBarsFromCSV("spy", "spy.csv")

feed = csvfeed.GenericBarFeed(frequency=Frequency.MINUTE)
feed.addBarsFromCSV("qqq", "qqq_data.csv")

# strategy = LongShortStrategy(feed, "qqq")
strategy = PredictorStrategy(feed, "qqq", DummyPredictor())
strategy.run()
portfolio_value = strategy.getBroker().getEquity() + strategy.getBroker().getCash()
print(portfolio_value)
