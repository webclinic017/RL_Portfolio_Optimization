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

strategy = LongShortStrategy(feed, "qqq")
strategy.run()
portfolio_value = strategy.getBroker().getEquity() + strategy.getBroker().getCash()
print(portfolio_value)
