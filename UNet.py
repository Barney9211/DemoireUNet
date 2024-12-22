import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2

class MoireDataset(Dataset):
    def __init__(self, moire_dir, clean_dir, transform=None):
        # moire_dir: 含有摩爾紋影像的資料夾
        # clean_dir: 對應乾淨影像的資料夾
        self.moire_dir = moire_dir
        self.clean_dir = clean_dir
        self.moire_files = sorted(os.listdir(moire_dir))
        self.clean_files = sorted(os.listdir(clean_dir))
        self.transform = transform

    def __len__(self):
        return len(self.moire_files)

    def __getitem__(self, idx):
        moire_path = os.path.join(self.moire_dir, self.moire_files[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_files[idx])

        moire_img = Image.open(moire_path).convert('RGB')
        clean_img = Image.open(clean_path).convert('RGB')

        if self.transform:
            moire_img = self.transform(moire_img)
            clean_img = self.transform(clean_img)



        return moire_img, clean_img
    

    
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 建立資料轉換
    transform = transforms.Compose([
        transforms.Resize((600,600)),
        transforms.ToTensor()
    ])
    


    # 建立資料集與 DataLoader
    moire_dir = "MoirePattenData/Moire"
    clean_dir = "MoirePattenData/Good"
    train_dataset = MoireDataset(moire_dir=moire_dir, clean_dir=clean_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # 建立模型與損失函數、優化器
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.L1Loss()  # 可依任務更改，如nn.MSELoss()、nn.L1Loss()、或自訂
    optimizer = optim.Adam(model.parameters(), lr=0.001)#0.001

    # 設定訓練迭代次數
    num_epochs = 5

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        a = tqdm(enumerate(train_loader),total= len(train_loader),desc=f"epoch:{epoch+1}/{num_epochs}")
        for i, (moire_img, clean_img) in a:            
            moire_img = moire_img.to(device)
            clean_img = clean_img.to(device)

            # 前向傳遞
            preds = model(moire_img)
            loss = criterion(preds, clean_img)

            # 反向傳遞與更新參數
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            a.set_postfix(loss=epoch_loss/(i+1))

        avg_loss = epoch_loss / len(train_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # torch.save(model.state_dict(), f"outputModel/unet_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), f"outputModel/outputModel.pth")
        

    print("訓練完成！")
