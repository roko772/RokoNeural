# RokoNeural strategy
# Stupid simple single layer NN strategy

# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import (DecimalParameter,
                                IStrategy, IntParameter)


# This class is a sample. Feel free to customize it.
class RokoNeuralV2(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_buy_trend, populate_sell_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 10000
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.9

    # Trailing stoploss
    trailing_stop = False
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Hyperoptable parameters
    # Buy neural network weights
    buy_w1_1 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w1_2 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w1_3 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w1_4 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w1_5 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w1_6 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)

    buy_w2_1 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w2_2 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w2_3 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w2_4 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w2_5 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w2_6 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)

    buy_w3_1 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w3_2 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w3_3 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w3_4 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w3_5 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w3_6 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)

    buy_w4_1 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w4_2 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w4_3 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w4_4 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w4_5 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w4_6 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)

    buy_w5_1 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w5_2 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w5_3 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w5_4 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w5_5 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_w5_6 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)

    buy_exit_1 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_exit_2 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_exit_3 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_exit_4 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)
    buy_exit_5 = DecimalParameter(low=-10, high=10, default=0, space='buy', optimize=True, load=True)

    buy_activation = IntParameter(low=0, high=10000, default=0, space='buy', optimize=True, load=True)

    # Sell neural network weights
    sell_w1_1 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w1_2 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w1_3 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w1_4 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w1_5 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w1_6 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)

    sell_w2_1 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w2_2 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w2_3 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w2_4 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w2_5 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w2_6 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)

    sell_w3_1 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w3_2 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w3_3 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w3_4 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w3_5 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w3_6 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)

    sell_w4_1 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w4_2 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w4_3 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w4_4 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w4_5 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w4_6 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)

    sell_w5_1 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w5_2 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w5_3 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w5_4 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w5_5 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_w5_6 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)

    sell_exit_1 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_exit_2 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_exit_3 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_exit_4 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_exit_5 = DecimalParameter(low=-10, high=10, default=0, space='sell', optimize=True, load=True)
    sell_activation = IntParameter(low=0, high=10000, default=0, space='sell', optimize=True, load=True)
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Momentum Indicators
        # ------------------------------------

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_norm'] = (dataframe['rsi'] - 50) * 2
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['mfi_norm'] = (dataframe['mfi'] - 50) * 2
        # Williams %R
        dataframe['willr'] = ta.WILLR(dataframe)
        dataframe['willr_norm'] = (dataframe['willr'] + 50) * 2
        # ROC
        dataframe['roc'] = ta.ROC(dataframe)

        dataframe['uo'] = ta.ULTOSC(dataframe)

        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        nn_input= [dataframe['cci'], dataframe['rsi_norm'], dataframe['mfi_norm'], dataframe['willr_norm'],
                   dataframe['uo'], 1]
        # print(nn_input)
        # print(len(nn_input))
        buy_weight_1 = [self.buy_w1_1.value, self.buy_w1_2.value, self.buy_w1_3.value, self.buy_w1_4.value,
                        self.buy_w1_5.value, self.buy_w1_6.value]
        buy_weight_2 = [self.buy_w2_1.value, self.buy_w2_2.value, self.buy_w2_3.value, self.buy_w2_4.value,
                        self.buy_w2_5.value, self.buy_w2_6.value]
        buy_weight_3 = [self.buy_w3_1.value, self.buy_w3_2.value, self.buy_w3_3.value, self.buy_w3_4.value,
                        self.buy_w3_5.value, self.buy_w3_6.value]
        buy_weight_4 = [self.buy_w4_1.value, self.buy_w4_2.value, self.buy_w4_3.value, self.buy_w4_4.value,
                        self.buy_w4_5.value, self.buy_w4_6.value]
        buy_weight_5 = [self.buy_w5_1.value, self.buy_w5_2.value, self.buy_w5_3.value, self.buy_w5_4.value,
                        self.buy_w5_5.value, self.buy_w5_6.value]
        buy_neuron_1, buy_neuron_2, buy_neuron_3, buy_neuron_4, buy_neuron_5 = 0, 0, 0, 0, 0

        for i in range(len(nn_input)):
            buy_neuron_1 += nn_input[i] * buy_weight_1[i]
            buy_neuron_2 += nn_input[i] * buy_weight_2[i]
            buy_neuron_3 += nn_input[i] * buy_weight_3[i]
            buy_neuron_4 += nn_input[i] * buy_weight_4[i]
            buy_neuron_5 += nn_input[i] * buy_weight_5[i]

        buy_exit_neuron = self.buy_exit_1.value * buy_neuron_1 + self.buy_exit_2.value * buy_neuron_2 + self.buy_exit_3.value * buy_neuron_3 + self.buy_exit_4.value * buy_neuron_4 + self.buy_exit_5.value * buy_neuron_5
        dataframe.loc[
            (
                    (buy_exit_neuron > self.buy_activation.value) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with sell column
        """
        nn_input = [dataframe['cci'], dataframe['rsi_norm'], dataframe['mfi_norm'], dataframe['willr_norm'],
                    dataframe['uo'], 1]
        # print(nn_input)
        # print(len(nn_input))
        sell_weight_1 = [self.sell_w1_1.value, self.sell_w1_2.value, self.sell_w1_3.value, self.sell_w1_4.value,
                         self.sell_w1_5.value, self.sell_w1_6.value]
        sell_weight_2 = [self.sell_w2_1.value, self.sell_w2_2.value, self.sell_w2_3.value, self.sell_w2_4.value,
                         self.sell_w2_5.value, self.sell_w2_6.value]
        sell_weight_3 = [self.sell_w3_1.value, self.sell_w3_2.value, self.sell_w3_3.value, self.sell_w3_4.value,
                         self.sell_w3_5.value, self.sell_w3_6.value]
        sell_weight_4 = [self.sell_w4_1.value, self.sell_w4_2.value, self.sell_w4_3.value, self.sell_w4_4.value,
                         self.sell_w4_5.value, self.sell_w4_6.value]
        sell_weight_5 = [self.sell_w5_1.value, self.sell_w5_2.value, self.sell_w5_3.value, self.sell_w5_4.value,
                         self.sell_w5_5.value, self.sell_w5_6.value]
        sell_neuron_1, sell_neuron_2, sell_neuron_3, sell_neuron_4, sell_neuron_5 = 0, 0, 0, 0, 0

        for i in range(len(nn_input)):
            sell_neuron_1 += nn_input[i] * sell_weight_1[i]
            sell_neuron_2 += nn_input[i] * sell_weight_2[i]
            sell_neuron_3 += nn_input[i] * sell_weight_3[i]
            sell_neuron_4 += nn_input[i] * sell_weight_4[i]
            sell_neuron_5 += nn_input[i] * sell_weight_5[i]
        sell_exit_neuron = self.sell_exit_1.value * sell_neuron_1 + self.sell_exit_2.value * sell_neuron_2 + self.sell_exit_3.value * sell_neuron_3 + self.sell_exit_4.value * sell_neuron_4 + self.sell_exit_5.value * sell_neuron_5
        # print("sell exit neuron : ", sell_exit_neuron)
        dataframe.loc[
            (
                    (sell_exit_neuron > self.sell_activation.value) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1
        return dataframe
