import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
#
# 사이드바 설정
st.sidebar.title('모델 설정')
model_option = st.sidebar.selectbox('모델 선택', ['모델 1', '모델 2'])
learning_rate = st.sidebar.slider('학습률', min_value=0.001, max_value=0.1, value=0.01)
epochs = st.sidebar.slider('에포크 수', min_value=1, max_value=20, value=5)
start_training = st.sidebar.button('학습 시작')

# 메인 화면
st.title('PyTorch MNIST 학습')

# 모델 정의
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 인스턴스 생성
if model_option == '모델 1':
    model = Model1()
else:
    model = Model2()

# MNIST 데이터셋 다운로드 및 로드
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 시작 버튼이 클릭되면 학습을 시작
if start_training:
    for epoch in range(epochs):  # 에포크 반복
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        st.write(f'에포크 {epoch + 1}, 손실: {running_loss / len(trainloader)}')
    st.success('학습 완료!')
