import os
import sys

from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

from fbprophet import Prophet
from resource import predict_image, analyze
from winreg import *

import pandas as pd
import io
import requests
import numpy as np

confirmed_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
                'csse_covid_19_data/csse_covid_19_time_series/' \
                'time_series_covid19_confirmed_global.csv'
deaths_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
             'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
recovered_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/' \
                'master/csse_covid_19_data/csse_covid_19_time_series/' \
                'time_series_covid19_recovered_global.csv'
countries_dataset_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/" \
                        "master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv"

df_confirmed = pd.read_csv(io.StringIO(requests.get(confirmed_url).content.decode('utf-8')))
df_deaths = pd.read_csv(io.StringIO(requests.get(deaths_url).content.decode('utf-8')))
df_recovered = pd.read_csv(io.StringIO(requests.get(recovered_url).content.decode('utf-8')))
countries = pd.read_csv(io.StringIO(requests.get(countries_dataset_url).content.decode('utf-8')))

population = countries[countries['Province_State'].isnull()][['Country_Region', 'Population']] \
    .rename(columns={'Country_Region': 'Country/Region'}).set_index('Country/Region')
infected = df_confirmed.groupby('Country/Region').sum().reset_index().set_index('Country/Region') \
    .join(population, on='Country/Region')
deaths = df_deaths.groupby('Country/Region').sum().reset_index().set_index('Country/Region') \
    .join(population, on='Country/Region')
recovered = df_recovered.groupby('Country/Region').sum().reset_index().set_index('Country/Region') \
    .join(population, on='Country/Region')


df_2 = []
for country in df_confirmed['Country/Region']:
    confirmed_cr = df_confirmed[df_confirmed['Country/Region'] == country]
    if len(confirmed_cr['Province/State'].values) > 1:
        for province in confirmed_cr['Province/State']:
            confirmed_ps = df_confirmed[df_confirmed['Province/State'] == province]
            confirmed_ps_ts = pd.DataFrame(confirmed_ps.iloc[:, 4:].sum()).reset_index()
            confirmed_ps_ts.columns = ['Date', 'Confirmed']
            confirmed_ps_ts['Province/State'] = province
            confirmed_ps_ts['Country/Region'] = confirmed_cr['Country/Region'].values[0]
            confirmed_ps_ts = confirmed_ps_ts[['Province/State', 'Country/Region',
                                               'Date', 'Confirmed']]
            confirmed_ps_ts['Death'] = pd.DataFrame(df_deaths[df_deaths['Province/State']
                                                              == province].iloc[:, 4:].sum())[
                0].values
            confirmed_ps_ts['Recovered'] = pd.DataFrame(df_recovered[df_recovered['Province/State']
                                                                     == province].iloc[:, 4:].sum())[
                0].values
            df_2.append(confirmed_ps_ts)
    else:
        confirmed_cr_ts = pd.DataFrame(confirmed_cr.iloc[:, 4:].sum()).reset_index()
        confirmed_cr_ts.columns = ['Date', 'Confirmed']
        confirmed_cr_ts['Province/State'] = None
        confirmed_cr_ts['Country/Region'] = confirmed_cr['Country/Region'].values[0]
        confirmed_cr_ts = confirmed_cr_ts[['Province/State', 'Country/Region',
                                           'Date', 'Confirmed']]
        confirmed_cr_ts['Death'] = pd.DataFrame(df_deaths[df_deaths['Country/Region']
                                                          == country].iloc[:, 4:].sum())[0].values
        confirmed_cr_ts['Recovered'] = pd.DataFrame(df_recovered[df_recovered['Country/Region']
                                                                 == country].iloc[:, 4:].sum())[
            0].values
        df_2.append(confirmed_cr_ts)

df_2 = pd.concat(df_2)
df_2['Date'] = pd.to_datetime(df_2['Date'])
df_2 = df_2.sort_values(by=['Date']).reset_index(drop=True)


class Preview_Dialog(QDialog):
    def __init__(self, parent, filename):
        super(Preview_Dialog, self).__init__(parent)
        uic.loadUi('dialog.ui', self)
        self.pixmap = QPixmap(filename).scaled(150, 150, Qt.KeepAspectRatio)
        self.label.setPixmap(self.pixmap)


class App(QMainWindow):
    mode = QueryValueEx(OpenKey(ConnectRegistry(None, HKEY_CURRENT_USER),
                                r'SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize'),
                        "AppsUseLightTheme")

    def __init__(self):
        super().__init__()
        uic.loadUi('design.ui', self)
        self.images = []
        self.pushButton.pressed.connect(self.add_folder)
        self.pushButton_add.pressed.connect(self.add_image)
        self.pushButton_remove.pressed.connect(self.remove_image)
        self.pushButton_predict.pressed.connect(self.predict_image)
        self.pushButton_clear.pressed.connect(self.clear_list)
        self.pushButton_2.pressed.connect(self.predict_all_images)
        
        self.actionOpen.triggered.connect(self.add_image)
        self.actionRemove.triggered.connect(self.remove_image)
        self.actionAdd_folder.triggered.connect(self.add_folder)
        self.setAcceptDrops(True)

        self.actionOpen.setShortcut('Ctrl+O')
        self.actionRemove.setShortcut('Ctrl+D')
        self.actionAdd_folder.setShortcut('Ctrl+Shift+O')

        self.plot_rt_forecast_for_country('Kazakhstan')
        self.pushButton_3.pressed.connect(lambda:
                                          self.plot_rt_forecast_for_country(self.lineEdit.text()))

    def add_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select a directory', '.')

        if dir_path != "":
            for i in [dir_path + '/' + file for file in os.listdir(dir_path)
                      if os.path.isfile(os.path.join(dir_path, file))]:
                self.listWidget.insertItem(self.listWidget.count(), i)

    def add_image(self):
        file_name = QFileDialog.getOpenFileName(self, 'Select an image', '.')
        if file_name[0] != "":
            self.open_dialog = Preview_Dialog(self, file_name[0])
            if self.open_dialog.exec_() == QDialog.Accepted:
                self.listWidget.insertItem(self.listWidget.count(), file_name[0])

    def remove_image(self):
        if self.listWidget.item(self.listWidget.currentRow()):
            get_reply = QMessageBox.question(self, "Remove An Image File", "Do you want to remove "
                                             + self.listWidget.currentItem().text()
                                             + " from the list?", QMessageBox.Yes | QMessageBox.No)
            if get_reply == QMessageBox.Yes:
                self.listWidget.takeItem(self.listWidget.currentRow())

    def predict_image(self):
        value = predict_image(self.listWidget.currentItem().text())
        self.result.setText(str(value))
        self.result.setStyleSheet(f'background-color:{"green" if value <= 0.5 else "red"};')
        self.listWidget.currentItem().setBackground(QColor("green" if value <= 0.5 else "red"))

    def clear_list(self):
        reply = QMessageBox.question(self, "Clear List Box", "Do you want to clear all the selected "
                                                             "images?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.listWidget.clear()

    def predict_all_images(self):
        if not self.listWidget.count():
            message = QMessageBox()
            message.setIcon(QMessageBox.Critical)
            message.setText("Error")
            message.setInformativeText("You haven't selescted images yet")
            message.setWindowTitle("Error")
            message.exec_()
        for index in range(self.listWidget.count()):
            item = self.listWidget.item(index)
            item.setBackground(QColor("green" if predict_image(item.text()) <= 0.5 else "red"))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            self.listWidget.insertItem(self.listWidget.count(), url.toString()[8:])

    def plot_rt_forecast_for_country(self, country_name):
        if country_name not in countries['Country_Region'].unique():
            message = QMessageBox()
            message.setIcon(QMessageBox.Critical)
            message.setText("Error")
            message.setInformativeText("Not valid country name")
            message.setWindowTitle("Error")
            message.exec_()
            return

        self.graphicsView = PlotWidget(self.tab_2)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout_4.addWidget(self.graphicsView, 2, 0, 1, 2)
        if self.mode:
            self.graphicsView.setBackground('w')
        self.graphicsView = PlotWidget(self.tab_2)
        self.gridLayout_4.addWidget(self.graphicsView, 2, 0, 1, 2)
        self.graphicsView_2 = PlotWidget(self.tab_2)
        self.gridLayout_4.addWidget(self.graphicsView_2, 2, 2, 1, 2)
        if self.mode:
            self.graphicsView.setBackground('w')
            self.graphicsView_2.setBackground('w')

        population, df = analyze(country_name, infected, recovered, deaths)
        start_date = df.iloc[0, 0]
        df['Days'] = df['Date'].apply(lambda x: (x - start_date).days)
        df['PI'] = df['Infected'] / population * 100
        df['Delta_Infected_Gr'] = (df['Infected'].diff()).clip(lower=0) / population * 1000000
        first_y_axe, second_y_axe = self.graphicsView.plotItem, pg.ViewBox()
        first_y_axe.showAxis('right')
        first_y_axe.scene().addItem(second_y_axe)
        first_y_axe.getAxis('right').linkToView(second_y_axe)
        second_y_axe.setXLink(first_y_axe)

        def updateViews(first_y_axe, second_y_axe):
            second_y_axe.setGeometry(first_y_axe.vb.sceneBoundingRect())
            second_y_axe.linkedViewChanged(first_y_axe.vb, second_y_axe.XAxis)

        first_y_axe.vb.sigResized.connect(lambda: updateViews(first_y_axe, second_y_axe))

        first_y_axe.plot([df['Days'].iloc[0], df['Days'].iloc[-1]], [1, 1],
                         pen=pg.mkPen(color=(255, 0, 0), width=1, style=Qt.DashLine))
        first_y_axe.plot(df['Rt'].dropna().tolist(),
                         pen=pg.mkPen(color=(255, 0, 0), width=3))

        second_y_axe.addItem(pg.PlotCurveItem(df['Delta_Infected_Gr']
                                              [:len(df['Rt'].dropna())].tolist(), pen='b', width=3))
        first_y_axe.setTitle(f"Population = {population},"
                             f" Infection start date (>100) = {start_date.date()}",
                             size="8pt")
        self.graphicsView.setLabel('left', 'Rt', **{'font-size': '14px'})
        self.graphicsView.setLabel('bottom', 'Days', **{'font-size': '14px'})
        self.graphicsView.setLabel('right', 'Infected (/day/1m)', **{'font-size': '14px'})
        self.graphicsView.showGrid(x=True, y=True)
        first_y_axe.setYRange(0, 8, padding=0)
        confirmed_country = df_2[df_2['Country/Region'] == 'Kazakhstan'].groupby(
            'Date').sum()['Confirmed'].reset_index()
        confirmed_country.columns = ['ds', 'y']
        confirmed_country['ds'] = pd.to_datetime(confirmed_country['ds'])
        model = Prophet(interval_width=0.95)
        model.fit(confirmed_country)
        forecast = model.predict(model.make_future_dataframe(periods=30))
        color = QColor('#0072B2')
        color.setAlphaF(0.2)
        yhat_lower = pg.PlotCurveItem(forecast['yhat_lower'].to_numpy(),
                                      pen=pg.mkPen(color=color))
        yhat_upper = pg.PlotCurveItem(forecast['yhat_upper'].to_numpy(),
                                      pen=pg.mkPen(color=color))
        yhat = pg.FillBetweenItem(yhat_lower, yhat_upper, brush=QBrush(color))
        self.graphicsView_2.addItem(yhat)
        self.graphicsView_2.addItem(yhat_lower)
        self.graphicsView_2.addItem(yhat_upper)
        self.graphicsView_2.plot(model.history['y'].to_numpy(),
                                 pen=pg.mkPen(color=QColor('black'), width=2, style=Qt.DotLine))
        self.graphicsView_2.plot(forecast['yhat'].to_numpy(),
                                 pen=pg.mkPen(color=QColor('#0072B2'), style=Qt.DashLine))
        self.graphicsView_2.showGrid(x=True, y=True)

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', 'Press Yes to Close.',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            qApp.quit()
        else:
            try:
                event.ignore()
            except AttributeError:
                pass


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.setStyle('Fusion')
    app.setWindowIcon(QIcon('icon.png'))

    if not ex.mode:
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.Active, QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, Qt.darkGray)
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, Qt.darkGray)
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, Qt.darkGray)
        dark_palette.setColor(QPalette.Disabled, QPalette.Light, QColor(53, 53, 53))

        app.setPalette(dark_palette)

        app.setStyleSheet(
            "QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")
    else:
        ex.graphicsView.setBackground('w')
        ex.graphicsView_2.setBackground('w')
    ex.show()
    sys.excepthook = except_hook
    sys.exit(app.exec_())
