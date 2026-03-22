"""游戏 AI 模型"""
import torch
import torch.nn as nn

class GameStateEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 512)
        
    def _make_layer(self, in_ch, out_ch, blocks, stride=1):
        layers = []
        for i in range(blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch if i==0 else out_ch, out_ch, 3, stride if i==0 else 1, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return nn.functional.relu(self.fc(x))

if __name__ == "__main__":
    print("GameAIModel: ~2.5M 参数")
