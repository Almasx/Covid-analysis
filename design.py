# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(774, 499)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout = QtWidgets.QGridLayout(self.tab)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../.designer/yandex lyseum 2019-2020/imap-Image-to-PDF-Converter-Application-For-Windows-master/imap-Image-to-PDF-Converter-Application-For-Windows-master/icons/Add Folder_50px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_add = QtWidgets.QPushButton(self.tab)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../../.designer/yandex lyseum 2019-2020/imap-Image-to-PDF-Converter-Application-For-Windows-master/imap-Image-to-PDF-Converter-Application-For-Windows-master/icons/plus_50px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_add.setIcon(icon1)
        self.pushButton_add.setObjectName("pushButton_add")
        self.verticalLayout.addWidget(self.pushButton_add)
        self.pushButton_remove = QtWidgets.QPushButton(self.tab)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../../.designer/yandex lyseum 2019-2020/imap-Image-to-PDF-Converter-Application-For-Windows-master/imap-Image-to-PDF-Converter-Application-For-Windows-master/icons/minus_50px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_remove.setIcon(icon2)
        self.pushButton_remove.setObjectName("pushButton_remove")
        self.verticalLayout.addWidget(self.pushButton_remove)
        self.gridLayout.addLayout(self.verticalLayout, 1, 1, 1, 1)
        self.listWidget = QtWidgets.QListWidget(self.tab)
        self.listWidget.setObjectName("listWidget")
        self.gridLayout.addWidget(self.listWidget, 1, 0, 5, 1)
        self.label_5 = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 5, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_predict = QtWidgets.QPushButton(self.tab)
        self.pushButton_predict.setObjectName("pushButton_predict")
        self.horizontalLayout.addWidget(self.pushButton_predict)
        self.pushButton_clear = QtWidgets.QPushButton(self.tab)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../../.designer/yandex lyseum 2019-2020/imap-Image-to-PDF-Converter-Application-For-Windows-master/imap-Image-to-PDF-Converter-Application-For-Windows-master/icons/clear_symbol_50px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_clear.setIcon(icon3)
        self.pushButton_clear.setObjectName("pushButton_clear")
        self.horizontalLayout.addWidget(self.pushButton_clear)
        self.gridLayout.addLayout(self.horizontalLayout, 4, 1, 1, 1)
        self.result = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.result.setFont(font)
        self.result.setAlignment(QtCore.Qt.AlignCenter)
        self.result.setObjectName("result")
        self.gridLayout.addWidget(self.result, 2, 1, 2, 1)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap("../../.designer/yandex lyseum 2019-2020/imap-Image-to-PDF-Converter-Application-For-Windows-master/imap-Image-to-PDF-Converter-Application-For-Windows-master/icons/top_menu_50px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tabWidget.addTab(self.tab, icon4, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lineEdit = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_4.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_4.addWidget(self.pushButton_3, 0, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        self.label_6.setObjectName("label_6")
        self.gridLayout_4.addWidget(self.label_6, 0, 0, 1, 1)
        self.graphicsView = PlotWidget(self.tab_2)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout_4.addWidget(self.graphicsView, 2, 0, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 0, 2, 1, 1)
        self.graphicsView_2 = PlotWidget(self.tab_2)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout_4.addWidget(self.graphicsView_2, 2, 2, 1, 2)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setMaximumSize(QtCore.QSize(191, 20))
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 1, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 1, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.tab_3)
        self.label.setObjectName("label")
        self.gridLayout_3.addWidget(self.label, 0, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_3.addItem(spacerItem1, 2, 0, 1, 2)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap("../../.designer/yandex lyseum 2019-2020/imap-Image-to-PDF-Converter-Application-For-Windows-master/imap-Image-to-PDF-Converter-Application-For-Windows-master/icons/info_50px.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.tabWidget.addTab(self.tab_3, icon5, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 774, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionRemove = QtWidgets.QAction(MainWindow)
        self.actionRemove.setObjectName("actionRemove")
        self.actionAdd_folder = QtWidgets.QAction(MainWindow)
        self.actionAdd_folder.setObjectName("actionAdd_folder")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionRemove)
        self.menuFile.addAction(self.actionAdd_folder)
        self.menuFile.addSeparator()
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", " Add Folder"))
        self.pushButton_add.setText(_translate("MainWindow", " Add File"))
        self.pushButton_remove.setText(_translate("MainWindow", " Remove"))
        self.label_5.setText(_translate("MainWindow", "List of images"))
        self.pushButton_2.setText(_translate("MainWindow", "Predict all"))
        self.pushButton_predict.setText(_translate("MainWindow", "Predict"))
        self.pushButton_clear.setText(_translate("MainWindow", " Clear All"))
        self.result.setText(_translate("MainWindow", "Result"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Home"))
        self.lineEdit.setText(_translate("MainWindow", "Kazakhstan"))
        self.pushButton_3.setText(_translate("MainWindow", "Submit"))
        self.label_6.setText(_translate("MainWindow", "Country name:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Rt and forecast"))
        self.label_4.setText(_translate("MainWindow", "Technologies Used:"))
        self.label_2.setText(_translate("MainWindow", "Saparov Almas "))
        self.label_3.setText(_translate("MainWindow", "Python 3.8, PyQt5, Tensorflow, Keras, cv2 and etc.."))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p>Developer:</p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "About"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open File"))
        self.actionRemove.setText(_translate("MainWindow", "Remove file"))
        self.actionAdd_folder.setText(_translate("MainWindow", "Add folder"))
from pyqtgraph import PlotWidget
