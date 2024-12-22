import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
import numpy as np

# 假設之前的 UNet 定義和相關設定均已存在...
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_channels=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels

        # Encoder
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for ch in feature_channels:
            self.downs.append(DoubleConv(prev_channels, ch))
            prev_channels = ch

        # Bottleneck
        self.bottleneck = DoubleConv(feature_channels[-1], feature_channels[-1]*2)

        # Decoder
        self.ups = nn.ModuleList()
        up_channels = feature_channels[-1]*2  # start from bottleneck (1024 if default)
        for ch in reversed(feature_channels):
            self.ups.append(nn.ConvTranspose2d(up_channels, ch, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(ch*2, ch))
            up_channels = ch

        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # upsample
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                x = nn.functional.pad(x, [diffX // 2, diffX - diffX//2,
                                          diffY // 2, diffY - diffY//2])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)


# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
model = UNet(in_channels=3, out_channels=3).to(device)
model.load_state_dict(torch.load("outputModel/outputModel.pth", map_location=device))
model.eval()

# 載入待處理的摩爾紋圖像 (使用 OpenCV 或 PIL)
# moire_image_path = "MoirePattenData/train/train/pair_00/0000_moire.jpg"
# moire_image_path = "MoirePattenData/TestMoire/0451_moire.jpg"
moire_image_path = "TestMoire/0448_moire.jpg"

# 使用 OpenCV 讀圖
image_bgr = cv2.imread(moire_image_path)
image_bgr=cv2.resize(image_bgr,(600,600))
cv2.imshow("origin",image_bgr)

if image_bgr is None:
    raise FileNotFoundError("Cannot load image: {}".format(moire_image_path))

# BGR轉RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 將影像轉為張量，並做同訓練時一致的縮放/正規化處理
input_tensor = TF.to_tensor(image_rgb)
input_tensor = TF.resize(input_tensor, (600, 600))
input_tensor = input_tensor.unsqueeze(0).to(device)  # shape: (1, 3, 256, 256)

# 前向傳遞
with torch.no_grad():
    preds = model(input_tensor)

# 將結果轉回CPU並去掉batch維度 (C,H,W)
output_tensor = preds.squeeze(0).cpu()

# 將張量轉為 NumPy 陣列，並轉換為 OpenCV 格式
output_numpy = output_tensor.permute(1, 2, 0).numpy()  # shape: (H, W, C)
output_numpy = np.clip(output_numpy, 0, 1)  # 確保值在 [0, 1] 範圍內
output_numpy = (output_numpy * 255).astype(np.uint8)  # 轉為 [0, 255] 範圍

# RGB轉BGR，因為OpenCV顯示需要BGR格式
output_bgr = cv2.cvtColor(output_numpy, cv2.COLOR_RGB2BGR)
output_bgr=cv2.resize(output_bgr,(600,600))

denoised_image = cv2.fastNlMeansDenoisingColored(output_bgr, None, 10, 10, 7, 21)
# denoised_image = cv2.GaussianBlur(output_bgr, (11, 11), 0)

cv2.imshow("non local means denoising",denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
