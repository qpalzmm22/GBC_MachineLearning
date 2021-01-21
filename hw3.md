# Autograd의 개념

`Neural Network`는 **1. Forward Propagation**, **2. Backward Propagation**으로 이루어져있다.

`Forward Propagation`은 주어진 Parameter로 최선의 결과는 짐작하는 것이고, `Backward Propagation`은 결과값과 실제값과의 차이의 loss function을 define하고 이의 `partial derivative`를 구해 각각의 노드의 parameter들을 조정한다.

![Pro](backPro.png)

`Foward pass`는 다음과 같이 이루어진다.
```python
import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
prediction = model(data) # forward pass
```

error는 결과값에서 실제값을 뺀값,
```python
loss = (prediction - labels).sum()
```

`Backward Propagation`은 다음과 같이 실행가능하다.
```python
loss.backward() # backward pass
```

`.backward()`가 실행이 되면 `.grad`에 `gradients`가 저장이 된다. 

`autograd`내에서는 `vector-jacobian product`가 실행된다.(실제로는 그것보다 더 efficient하고 간단하게 작동한다고 한다.)

![jacobian](jacobian.png)

Autograd에서는 이렇게 계산된 정보를 `Directed Acyclic Graph(DAG)`에 저장하는데 leaves(아래에서 파랑색 박스)는 input tensor 이고 roots는 output tensor이다. 

![dag_autograd](dag_autograd.png)

>DAG는  `.backward()`가 불릴 때마다 다시 만든다.그래서 iteration마다 연산의 크기와 모양을 바꿀 수 있다.

tensor는 `.requires_grad` flag가 True로 설정이 되면 gradient 연산을 필요로하고 False로 하면 제외가 된다.
이렇게 제외가 된 parameter를 **Frozen parameter**라고 부른다.

이렇게 `'Freezing'`하는 이유는 이렇게 freeze하는 parameter가 변하지 않았으면 할때 쓰는데 이는 주로 `transfer leaning`을 할 때 주로 사용된다. 연산을 모든 node에 대해서 하지 않기 때문에 `performance`도 좋아진다. 

# Neural Net의 개념

torch.nn package에서 Neural Network가 제공된다.

Neural Network에서의 훈련과정은 다음과 같다.
<b>
1. Neural Network에서 parameter가 무엇이 될것인지 결정한다.
1. input dataset을 여러번에 나눈다.
1. Network를 통과시키며 `Forward Propagation`을 진행한다.
1. 실제값과 결과값을 비교한다.
1. 마지막 layer에서 부터 연산을 역으로 진행시킨다.
1. 변화량과 각각 parameter의 비중을 계산하며 parameter들을 loss를 줄일 수 있는 방향으로 조정한다.
</b>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

위의 network에서 쓰인 함수들을 하나하나 해석해보자.

`nn.Conv2d(in_channels, out_channels, kernal_size )` # 2d Convolution을 적용시킨다.

`nn.Linear(in_feature, out_feature, bias)` # y = xA + b를 따르는 linear Transform을 적용시킨다. 

`x.view(*shape)` # tensor의 크기(size)나 모양(shape)을 변경. -1 이라면 다른 dimension을 보고 추론한다.

`F.relu()` # activation function, ReLU를 쓰겠다는 뜻.

`F.max_pool2d(kernel_size, stride=kernel_size)` # max pool을 kernel_size만큼 적용시킨다. 




그렇기 때문에 간단히 정리를 하자면 다음과 같다.
```

input -> conv2d(1,6,3) -> relu() -> maxpool2d(stride=(2,2)) -> conv2d(6, 16, 3) -> 
relu() -> maxpool2d(stride=2)
-> view -> linear(16 * 6 * 6, 120) -> relu() -> linear(120, 84) -> relu() -> linear(84, 10) 
-> output
```

이렇게 하면 foward 함수가 정의가 되는데, 자동으로 backward 함수가 적용이 된다. 예를 들어 `net.parameters()`를 하면 learnable parameter가 반환된다. 

`Loss function`은 `nn package` 밑에 있는데, 그 종류로는 MSE, L1, L2 loss 등이 있다.

backpropagation을 실행시키기 위해서는 그저 `.backward()`를 실행시키면 **이미 forward propagation에서 define**이 되있기 떄문에 알아서 계산을 해서 .grad 에 값이 저장이 된다.

`torch.optim`은 SGD, ADAM 등 Gradient Descent algorithm들을 가지고 있다. 
이를 이용해서 새로운 parameter값을 채워넣을 수 있다.
아래코드에서 사용한 SGD 는 다음의 formula를 사용한다.
```
weight = weight - learning_rate * gradient
```

```python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```



