import torch
torch.manual_seed(0)


x = torch.randn(10,4, requires_grad=True)
W = torch.randn(4,4, requires_grad=True)
y = torch.randn(10,4, requires_grad=True)

#defination of f
z = x.mm(W)
y_hat = torch.clamp(z, 0)
f = (y_hat-y).pow(2).sum()
f.backward()

#W_grad
y_hat_grad = (y_hat > 0).int()
z_grad = x.t()
W_grad = z_grad.mm((2 * (y_hat - y) * y_hat_grad))
print(W_grad, W.grad)

#X_grad
y_hat_grad = (y_hat > 0).int()
z_grad = W.t()
x_grad = (2 * (y_hat - y) * y_hat_grad).mm(z_grad)
print(x_grad, x.grad)

#y_grad
y_grad = (2 * (y_hat - y)) * -1
print(y_grad, y.grad)


