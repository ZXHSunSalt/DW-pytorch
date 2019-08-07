#solve the problem of regression: the number of sample is 64, input layers 1000 dim,# hidden layer 100, output layer 10dim
import torch

num_samples = 64  # N
dim_in, dim_hid, dim_out = 1000, 100, 10  # IN H OUT
x = torch.randn(num_samples, dim_in)  # N * IN
y = torch.randn(num_samples, dim_out)  # N * OUT

# 提前定义模型
model = torch.nn.Sequential(
    torch.nn.Linear(dim_in, dim_hid, bias=False),  # model[0]
    torch.nn.ReLU(),
    torch.nn.Linear(dim_hid, dim_out, bias=False),  # model[2]
)

# torch.nn.init.normal_(model[0].weight) #修改一
# torch.nn.init.normal_(model[2].weight)

# 提前定义loss函数和优化函数
loss_fun = torch.nn.MSELoss(reduction='sum')
eta = 1e-4  # 修改二
optimizer = torch.optim.Adam(model.parameters(), lr=eta)

for i in range(1000):
    # Forward pass
    y_pred = model(x)

    # Loss
    loss = loss_fun(y_pred, y)
    print(i, loss.item())

    optimizer.zero_grad()
    # Backward pass
    loss.backward()

    # update model parameters
    optimizer.step()