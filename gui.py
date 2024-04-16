#!/usr/bin/env python

import os, sys, time
from datetime import datetime as dt
import numpy as np
import pandas as pd
from scipy import interpolate
from indicators import *
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import pyqtSignal,Qt
from PyQt5.QtWidgets import QApplication,\
                            QPushButton,\
                            QWidget,\
                            QHBoxLayout,\
                            QVBoxLayout,\
                            QGridLayout,\
                            QLabel,\
                            QLineEdit,\
                            QTabWidget,\
                            QTabBar,\
                            QGroupBox,\
                            QDialog,\
                            QTableWidget,\
                            QTableWidgetItem,\
                            QInputDialog,\
                            QMessageBox,\
                            QComboBox,\
                            QShortcut,\
                            QFileDialog,\
                            QCheckBox,\
                            QRadioButton,\
                            QHeaderView,\
                            QSlider,\
                            QSpinBox,\
                            QDoubleSpinBox

try:
    data_path = sys.argv[1]
except:
    data_path = 'sample_data/data_BTCUSD_1.csv'

seed_money = 1000
n1, n2, n3 = 10, 20, 10 # MACD parameters
n4 = 20                 # Bollinger parameter
n5 = 15                 # RSI parameter
n6, n7, n8 = 15, 5, 3   # Stochastic parameters

intervals = ['1', '10', '60', '360']

class Widget(QDialog):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)
        self.originalPalette = QApplication.palette()
    
        # Load data
        self.df0 = pd.read_csv(data_path)
        self.interval_data = int(self.df0['timestamp'].diff()[1] // 60)
        self.setIndicators()

        # Top layout
        topLayout = QHBoxLayout()
        
        self.feeBox = QDoubleSpinBox()
        self.feeBox.setMinimum(0); self.feeBox.setMaximum(100); self.feeBox.setValue(0)

        self.lengthBox = QSpinBox()
        self.lengthBox.setMinimum(1); self.lengthBox.setMaximum(200); self.lengthBox.setValue(100)
        self.replotButton = QPushButton('Replot')
        self.replotButton.clicked.connect(self.replotChartBox)

        self.intervalBox = QSpinBox()
        self.intervalBox.setMinimum(1); self.intervalBox.setMaximum(1440); self.intervalBox.setValue(self.interval_data)
        self.refreshButton = QPushButton('Change interval')
        self.refreshButton.clicked.connect(self.refresh)

        topLayout.addWidget(QLabel('Fee [%]:'))
        topLayout.addWidget(self.feeBox)
        topLayout.addWidget(QLabel('Display length:'))
        topLayout.addWidget(self.lengthBox)
        topLayout.addWidget(self.replotButton)
        topLayout.addWidget(QLabel('Interval [minute]:'))
        topLayout.addWidget(self.intervalBox)
        topLayout.addWidget(self.refreshButton)
        
        self.initialize()

        # Middle layout
        self.fig, self.axs = plt.subplots(4, 1, sharex = True, squeeze = False, figsize = (10, 10))
        self.ax2 = self.axs[0, 0].twinx()
        self.plotChartBox()
        self.createOrderBox()

        # Bottom layout
        self.printButton = QPushButton('Print'); self.printButton.clicked.connect(self.printOutput)
        self.resetButton = QPushButton('Random reset date'); self.resetButton.clicked.connect(self.resetAll)

        # Main layout
        self.mainLayout = QGridLayout()
        self.mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        self.mainLayout.addWidget(self.chartBox, 1, 0); self.mainLayout.addWidget(self.orderBox, 1, 1)
        self.mainLayout.addWidget(self.printButton, 2, 0); self.mainLayout.addWidget(self.resetButton, 2, 1)
        self.setLayout(self.mainLayout)
        
        self.setWindowTitle(f'VTradeGUI (with data: {data_path})')

    def setIndicators(self, df = None):
        self.df = self.df0.copy() if type(df) == type(None) else df
        self.dohlc = self.df[['timestamp', 'open', 'high', 'low', 'close']].values
        self.ma5 = moving_average(self.df['close'], 5)
        self.ma10 = moving_average(self.df['close'], 10)
        self.ma30 = moving_average(self.df['close'], 30)
        self.ma60 = moving_average(self.df['close'], 60)
        self.macd = macd(self.df['close'], n1, n2)
        self.macd_signal = moving_average(self.macd, n3)
        self.macd_oscillator = self.macd - self.macd_signal
        self.boll_hi, self.boll_lo = bollinger(self.df['close'], n = n4, s = 2)
        self.rsi = get_rsi(self.df['close'], n = n5)
        self.stok, self.stod = sto_slow(self.df['close'], self.df['low'], self.df['high'], n6, n7, n8)

    def initialize(self):
        self.equity = seed_money
        self.position, self.bought, self.sold = 0, 0, 0
        #self.idx = np.random.randint(self.lengthBox.value(), len(self.df) - self.lengthBox.value())
        self.idx = np.random.randint(14400, len(self.df) - self.lengthBox.value()) # Min pred buffer 14400
        self.start = self.idx
        self.profits = [0]
        self.intervals = intervals

    def refresh(self):
        ts_ = self.df['timestamp'][self.idx]
        jump = self.intervalBox.value() // self.interval_data
        e = jump * (len(self.df0) // jump)
        ts = self.df0['timestamp'].values[jump - 1:e:jump]
        o = self.df0['open'].values[:e:jump]
        h = np.max(self.df0['high'][:e].values.reshape(len(self.df0) // jump, jump), axis = 1)
        l = np.min(self.df0['low'][:e].values.reshape(len(self.df0) // jump, jump), axis = 1)
        c = self.df0['close'].values[jump - 1:e:jump]
        v = np.sum(self.df0['volume'][:e].values.reshape(len(self.df0) // jump, jump), axis = 1)
        
        self.setIndicators(df = pd.DataFrame({'timestamp': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v}))
        try:
            self.idx = np.arange(len(self.df))[self.df['timestamp'] < ts_][-1] + 1
            self.start = self.idx
            self.replotChartBox()
        except:
            self.resetAll()

    def plotChartBox(self):
        self.chartBox = QGroupBox("Chart Box")

        self.plotChart()
        self.canvas = FigureCanvas(self.fig)

        self.layout = QGridLayout()
        self.layout.addWidget(self.canvas)
        self.chartBox.setLayout(self.layout)

    def replotChartBox(self):
        self.plotChart()
        self.canvas.draw()

    def createOrderBox(self):
        self.orderBox = QGroupBox("Order Box")
        layout = QGridLayout()
        
        self.qtyBox = QDoubleSpinBox(); self.qtyBox.setMinimum(0); self.qtyBox.setValue(0.01)
        self.buyButton = QPushButton('Buy (Long)'); self.buyButton.clicked.connect(self.buy)
        self.sellButton = QPushButton('Sell (Short)'); self.sellButton.clicked.connect(self.sell)
        self.stepButton = QPushButton('▶ Step ▶'); self.stepButton.clicked.connect(self.step)
        self.positionLabel = QLabel(f'Equity: {self.equity:.5f} $ \nBought: {self.bought:.2f} \nSold: {self.sold:.2f}')
        self.optBox = QComboBox()
        self.optBox.addItem('1m')
        self.optBox.addItem('10m')
        self.optBox.addItem('60m')
        self.optBox.addItem('360m')
        self.iterBox = QSpinBox(); self.iterBox.setMaximum(200); self.iterBox.setValue(10)
        #self.ATButton = QPushButton('Auto trade'); self.ATButton.clicked.connect(self.autoTrade)
        self.gotoButton = QPushButton('Go to date'); self.gotoButton.clicked.connect(self.goto)
        self.yrBox, self.mtBox, self.dyBox, self.hrBox, self.mnBox = QSpinBox(), QSpinBox(), QSpinBox(), QSpinBox(), QSpinBox()
        self.yrBox.setMaximum(3000)

        layout.addWidget(QLabel('Quantity'), 0, 0); layout.addWidget(self.qtyBox, 0, 1)
        layout.addWidget(self.buyButton, 1, 0, 1, 2)
        layout.addWidget(self.sellButton, 2, 0, 1, 2)
        layout.addWidget(self.stepButton, 3, 0, 1, 2)
        layout.addWidget(self.positionLabel, 4, 0)
        #layout.addWidget(QLabel('--------Prediction--------'), 5, 0, 1, 2)
        #layout.addWidget(self.predictLabel, 6, 0)
        #layout.addWidget(QLabel('AutoTrade option:'), 7, 0); layout.addWidget(self.optBox, 7, 1)
        #layout.addWidget(QLabel('AutoTrade steps:'), 8, 0); layout.addWidget(self.iterBox, 8, 1)
        #layout.addWidget(self.ATButton, 9, 0, 1, 2)
        layout.addWidget(QLabel('--------Time travel--------'), 10, 0, 1, 2)
        layout.addWidget(QLabel('Year:'), 11, 0); layout.addWidget(self.yrBox, 11, 1)
        layout.addWidget(QLabel('Month:'), 12, 0); layout.addWidget(self.mtBox, 12, 1)
        layout.addWidget(QLabel('Day:'), 13, 0); layout.addWidget(self.dyBox, 13, 1)
        layout.addWidget(QLabel('Hour:'), 14, 0); layout.addWidget(self.hrBox, 14, 1)
        layout.addWidget(QLabel('Minute:'), 15, 0); layout.addWidget(self.mnBox, 15, 1)
        layout.addWidget(self.gotoButton, 16, 0, 1, 2)

        self.orderBox.setLayout(layout)

    def updateOrderBox(self, predict=True):
        self.positionLabel.setText(f'Equity: {self.equity:.5f} $ \nBought: {self.bought:.2f} \nSold: {self.sold:.2f}')

    def printOutput(self):
        print(f"DateTime: {str(dt.fromtimestamp(self.df['timestamp'][self.idx]))}")
        print(f"Price: {self.df['close'][self.idx]} $")
        print(f"My profit: {self.profits[-1]} %")

    def resetAll(self):
        self.initialize()
        self.updateOrderBox()
        self.replotChartBox()

    def goto(self):
        self.initialize()
        ts = dt(self.yrBox.value(), self.mtBox.value(), self.dyBox.value(), self.hrBox.value(), self.mnBox.value()).timestamp()
        try:
            self.idx = np.arange(len(self.df))[self.df['timestamp'] < ts][-1] + 1
        except:
            self.idx = self.lengthBox.value()
        self.start = self.idx
        self.updateOrderBox()
        self.replotChartBox()

    def plotChart(self):
        s, e = self.idx + 1 - self.lengthBox.value(), self.idx + 1
        x = np.arange(-self.lengthBox.value() + 1, 1)
        xlim = [-self.lengthBox.value() - 1, 1]
        self.dohlc[s:e, 0] = x

        for i in range(4): self.axs[i, 0].clear()
        self.ax2.clear()

        self.axs[0, 0].set_title(str(dt.fromtimestamp(self.df['timestamp'][self.idx])))
        self.axs[0, 0].fill_between(x, self.boll_lo[s:e], self.boll_hi[s:e], alpha = 0.2, color = 'orange', label = 'Bollinger')
        self.axs[0, 0].plot(x, self.ma5[s:e], 'k', linewidth = 0.3, label = 'MA5')
        self.axs[0, 0].plot(x, self.ma10[s:e], 'b', linewidth = 0.3, label = 'MA10')
        self.axs[0, 0].plot(x, self.ma30[s:e], 'purple', linewidth = 0.3, label = 'MA30')
        self.axs[0, 0].plot(x, self.ma60[s:e], 'm', linewidth = 0.3, label = 'MA60')
        candlestick_ohlc(self.axs[0, 0], self.dohlc[s:e], width = 0.8, colorup = 'g', colordown = 'r')
        #self.axs[0, 0].axvline(x = 0, color = 'b', linestyle = '--')
        self.axs[0, 0].set_xlim(xlim)
        self.axs[0, 0].set_ylabel('Price [$]')
        self.axs[0, 0].legend(loc='upper left')
        self.axs[0, 0].xaxis.label.set_color('w')
        self.ax2.bar(x, self.df['volume'][s:e], color = 'gray', alpha = 0.3, label = 'Volume')
        self.ax2.tick_params(axis = 'y', labelcolor = 'gray')
        self.ax2.set_ylim([0, 3 * max(self.df['volume'][s:e])])
        self.ax2.set_ylabel('Volume', color = 'gray')

        self.axs[1, 0].bar(x, self.macd_oscillator[s:e], color = 'k', label = 'Oscillator')
        self.axs[1, 0].plot(x, self.macd[s:e], 'b', label = 'MACD')
        self.axs[1, 0].plot(x, self.macd_signal[s:e], 'r', label = 'Signal')
        self.axs[1, 0].axvline(x = 0, color = 'b', linestyle = '--')
        self.axs[1, 0].set_xlim(xlim)
        self.axs[1, 0].set_ylabel('MACD')
        self.axs[1, 0].legend(loc='upper left')
        self.axs[1, 0].xaxis.label.set_color('w')

        self.axs[2, 0].plot(x, self.rsi[s:e], 'g', label = 'RSI')
        self.axs[2, 0].plot(x, self.stok[s:e], 'b', label = 'Sto_K')
        self.axs[2, 0].plot(x, self.stod[s:e], 'r', label = 'Sto_D')
        self.axs[2, 0].axhline(y=80, color='k', linestyle='--')
        self.axs[2, 0].axhline(y=20, color='k', linestyle='--')
        self.axs[2, 0].axvline(x = 0, color = 'b', linestyle = '--')
        self.axs[2, 0].set_xlim(xlim)
        self.axs[2, 0].set_ylim([0, 100])
        self.axs[2, 0].set_ylabel('Momentum [%]')
        self.axs[2, 0].legend(loc='upper left')
        self.axs[2, 0].xaxis.label.set_color('w')

        self.axs[3, 0].plot(x[-(self.idx - self.start) - 1:], 100 * (self.df['close'].values[s:e][-(self.idx - self.start) - 1:] / self.df['close'].values[self.start] - 1), 'k--', label = 'Buy and hold')
        self.axs[3, 0].plot(x[-(self.idx - self.start) - 1:], self.profits[-self.lengthBox.value():][-(self.idx - self.start) - 1:], 'b', label = 'My profit')
        self.axs[3, 0].axvline(x = 0, color = 'b', linestyle = '--')
        self.axs[3, 0].set_xlim(xlim)
        self.axs[3, 0].set_xlabel('Relative time index')
        self.axs[3, 0].set_ylabel('Profit [%]')
        self.axs[3, 0].legend(loc='upper left')

    def step(self):
        if self.idx == len(self.df) - 1:
            self.initialize()
        else:
            self.idx += 1
            self.equity += self.position * (self.df['close'][self.idx] - self.df['close'][self.idx - 1])
            self.profits.append((self.equity - seed_money) / seed_money * 100)
            self.updateOrderBox()
        
        self.replotChartBox()

    def buy(self):
        self.position += self.qtyBox.value()
        self.bought, self.sold = (self.position, 0) if self.position > 0 else (0, -self.position)
        self.equity -= self.qtyBox.value() * self.df['close'][self.idx] * self.feeBox.value() / 100
        self.updateOrderBox(predict=False)

    def sell(self):
        self.position -= self.qtyBox.value()
        self.bought, self.sold = (self.position, 0) if self.position > 0 else (0, -self.position)
        self.equity -= self.qtyBox.value() * self.df['close'][self.idx] * self.feeBox.value() / 100
        self.updateOrderBox(predict=False)


if __name__ == '__main__':
    app = QApplication([])
    window = Widget()
    window.show()
    app.exec()
