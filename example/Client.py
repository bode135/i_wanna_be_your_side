#!/usr/bin/python3
# 文件名：client.py

# 导入 socket、sys 模块
import socket
import sys

# 创建 socket 对象
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 获取本地主机名
host = socket.gethostname()

# 设置端口号
port = 9999

# 连接服务，指定主机和端口
s.connect((host, port))

# 接收小于 1024 字节的数据
msg = s.recv(1024)
msg
s.close()

print (msg.decode('utf-8'))

msg = ''
# python C:\Users\Administrator\Desktop\torch_policity\test.py

# cd C:\Users\Administrator\Desktop\torch_policity

# C:\Users\Administrator\AppData\Local\Programs\Python\Python38-321\python.exe test.py
# C:\Users\Administrator\AppData\Local\Programs\Python\Python38-321\python.exe test1.py