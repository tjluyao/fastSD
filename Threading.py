import threading
import time

inputs = []
def collect_input():
    while True:
        user_input = input()
        inputs.append(user_input)


def output_operation():
    while True:
        # 执行其他操作，可以根据需要进行定义
        print('Print:')
        for input in inputs:
            print(input)
        time.sleep(5)

# 创建收集输入的线程
input_thread = threading.Thread(target=collect_input)
input_thread.daemon = True  # 设置为守护线程，当主线程退出时自动退出

output_thread = threading.Thread(target=output_operation)

# 启动线程
input_thread.start()
output_thread.start()

# 主线程继续执行其他操作
while True:
    # 执行主线程的操作

    time.sleep(2)