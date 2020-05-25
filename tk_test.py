import tkinter as tk
import tkinter.filedialog
from sensor_fault import *

class my_gui():
    def __init__(self, init_windows):
        self.init_windows = init_windows

    def set_init_windows(self):
        self.init_windows.title('传感器故障诊断')
        self.center_window(620, 300)
        self.init_windows.lb = tk.Label(windows, borderwidth=1, relief='solid')
        self.init_windows.get_path_button = tk.Button(
            windows, text='获取预测数据集', command=self.get_path)
        self.init_windows.get_path_button.place(x=260, y=10)
        self.init_windows.get_path_button = tk.Button(
            windows, text='DNN下原数据图像', command=self.DNN_draw)
        self.init_windows.get_path_button.place(x=160, y=130)
        self.init_windows.get_path_button = tk.Button(
            windows, text='DNN诊断结果', command=self.DNN_pre)
        self.init_windows.get_path_button.place(x=370, y=130)
        self.init_windows.get_path_button = tk.Button(
            windows, text='LSTM-RNN下原数据图像', command=self.LSTM_draw)
        self.init_windows.get_path_button.place(x=140, y=220)
        self.init_windows.get_path_button = tk.Button(
            windows, text='LSTM-RNN诊断结果', command=self.LSTM_pre)
        self.init_windows.get_path_button.place(x=360, y=220)

    # 居中显示窗口
    def center_window(self, width=300, height=200):
        # get screen width and height
        screen_width = self.init_windows.winfo_screenwidth()
        screen_height = self.init_windows.winfo_screenheight()
        # calculate position x and y coordinates
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        self.init_windows.geometry('%dx%d+%d+%d' % (width, height, x, y))

    # 获取数据集路径
    def get_path(self):
        self.filename = tk.filedialog.askopenfilename()
        self.init_windows.lb.place(x=10, y=45)
        # print(filename)
        if self.filename != '':
            self.init_windows.lb.config(text="您选择的文件是："+self.filename)
        else:
            self.init_windows.lb.config(text="您没有选择任何文件")
    def DNN_draw(self):
        self.sensor_type = {'Sensor1': '无故障', 'Sensor2': '无故障',
                   'Sensor3': '无故障', 'Sensor4': '无故障'}
        self.result, self.actual = pre_DNN(self.filename)
        draw_picture_DNN(self.result, self.actual, self.sensor_type)
    def DNN_pre(self):
        self.sensor_type['Sensor1'] = fault(self.result, self.actual, 0)
        self.sensor_type['Sensor2'] = fault(self.result, self.actual, 1)
        self.sensor_type['Sensor3'] = fault(self.result, self.actual, 2)
        self.sensor_type['Sensor4'] = fault(self.result, self.actual, 3)
        self.actual = fault_insertion(self.actual)
        draw_picture_DNN(self.result, self.actual, self.sensor_type)
    def LSTM_draw(self):
        self.sensor_type = {'Sensor1': '无故障', 'Sensor2': '无故障',
                   'Sensor3': '无故障', 'Sensor4': '无故障'}
        self.result, self.actual = pre_LSTMRNN(self.filename)
        draw_picture_LSTMRNN(self.result, self.actual, self.sensor_type)
    def LSTM_pre(self):
        self.sensor_type['Sensor1'] = fault(self.result, self.actual, 0)
        self.sensor_type['Sensor2'] = fault(self.result, self.actual, 1)
        self.sensor_type['Sensor3'] = fault(self.result, self.actual, 2)
        self.sensor_type['Sensor4'] = fault(self.result, self.actual, 3)
        self.actual = fault_insertion(self.actual)
        draw_picture_LSTMRNN(self.result, self.actual, self.sensor_type)



if __name__ == '__main__':
    windows = tk.Tk()
    my_win = my_gui(windows)
    my_win.set_init_windows()
    windows.mainloop()
