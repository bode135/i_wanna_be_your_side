import torch
x = torch.randn(2,requires_grad=True)
x = torch.FloatTensor([1,2])
x.requires_grad = True
y = torch.softmax(x,0)
y.retain_grad()
y_real = torch.IntTensor([0,1])

loss = y_real[0]*torch.log(y[0])+y_real[1]*torch.log(y[1])
loss.backward()  # 自动求导

print(loss)
print(y)
print(x)

# -1/y[1]           # d(loss)/d(y1) == y.grad[1]
# y[1] * (1 - y[1])  # d(y1)/d(x1)
# y[1]-1            # d(loss)/d(x1) == x.grad[1]
# -1/y[1]*y[1]*( 1 - y[1] )     # d(loss)/d(x)


print(y.grad)  # 求对x的梯度
print(x.grad)  # 求对x的梯度
