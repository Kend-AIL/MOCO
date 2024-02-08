import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

from model.model import RealValCNNLineDetNew
from model.loss import  CrossEntropyAcrossLines,WeightedCrossEntropyAcrossLines
from model.evaluation import evaluate
from model.data import Corruptdata
from tqdm import tqdm
def train_model(args):
    # 参数设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs
    eval_interval = args.eval_interval
    batch_size = args.batch_size
    lr = args.lr
    resume_training = args.resume_training
    checkpoint_path = args.checkpoint_path

    # 模型、数据集和优化器初始化
    model =RealValCNNLineDetNew().to(device)
    train_list=['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010',
                      'P011', 'P012', 'P013', 'P014', 'P015', 'P016', 'P017', 'P018', 'P019', 'P020',
                      'P021', 'P022', 'P023', 'P024', 'P025', 'P026', 'P027', 'P028', 'P029', 'P030',
                      'P031', 'P032', 'P033', 'P034', 'P035', 'P036', 'P037', 'P038', 'P039', 'P040',
                      'P041', 'P042', 'P043', 'P044', 'P045', 'P046', 'P047', 'P048', 'P049', 'P050',
                      'P051', 'P052', 'P053', 'P054', 'P055', 'P056', 'P057', 'P058', 'P059', 'P060',
                      'P061', 'P062', 'P063', 'P064', 'P065', 'P066', 'P067', 'P068', 'P069', 'P070',
                      'P071', 'P072', 'P073', 'P074', 'P075', 'P076', 'P077', 'P078', 'P079', 'P080',
                      'P081', 'P082', 'P083', 'P084', 'P085', 'P086', 'P087', 'P088', 'P089', 'P090',
                      'P091', 'P092', 'P093', 'P094', 'P095', 'P096', 'P097', 'P098', 'P099', 'P100',
                      'P101','P102','P103','P104','P105','P106','P107','P108','P109','P110']
    eval_list=['P111', 'P112', 'P113', 'P114', 'P115', 'P116', 'P117', 'P118', 'P119', 'P120',]
    window_size=4
    dir= '/mnt/datasets/CMR/MICCAIChallenge2023/ChallengeData/SingleCoil/Cine/PD/train/TrainingSet'
    mask_dir='/mnt/datasets/CMR/MICCAIChallenge2023/ChallengeData/SingleCoil/Cine/PD/train/TrainingSet/Noise_Mask_6x8x4'
    train_dataset = Corruptdata(dir=dir,mask_dir=mask_dir,window_size=window_size,patient_list=train_list)
    eval_dataset=Corruptdata(dir=dir,mask_dir=mask_dir,window_size=window_size,patient_list=eval_list)
    optimizer = optim.Adam(model.parameters(), lr=lr,betas=[0.9,0.99])

    # 损失函数和评估函数
    criterion = WeightedCrossEntropyAcrossLines(weight_cl0=4)
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,prefetch_factor=2,num_workers=4)
    eval_loader=DataLoader(eval_dataset, batch_size=batch_size, shuffle=True,prefetch_factor=2,num_workers=4)
    # Tensorboard SummaryWriter
    writer = SummaryWriter()

    # 加载已保存的模型参数
    if resume_training and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        epoch = 0

    # 训练循环
    for epoch in range(epoch, epochs):
        model.train()
        running_loss = 0.0
        i=0
        for inputs,_,labels in tqdm(train_loader):
            inputs=inputs.to('cuda')
            labels=labels.to('cuda')
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.to(torch.float32))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                writer.add_scalar('training_loss', running_loss / 100, epoch * len(train_loader) + i)
                print(f'training_loss:{running_loss/100}')
                running_loss = 0.0
            i=i+1

        # 每隔 eval_interval 轮进行一次评估
        if epoch % eval_interval == 0 and epoch!=0:
            model.eval()
            print('evaluation begin...')
            with torch.no_grad():
                for inputs,_,labels in tqdm(eval_loader):
                    inputs=inputs.to('cuda')
                    labels=labels.to('cuda')
                    outputs = model(inputs)
                    acc,pre,rec,f1=evaluate(outputs,labels.float())
                    writer.add_scalar('accuracy', acc, epoch)
                    writer.add_scalar('precision', pre, epoch)
                    writer.add_scalar('recall', rec, epoch)
                    writer.add_scalar('f1_score', f1, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, checkpoint_path+f'_{epoch}')

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch 模型训练脚本')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--eval_interval', type=int, default=10, help='eval 间隔')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--resume_training', action='store_true', help='是否恢复训练')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='模型检查点路径')
    args = parser.parse_args()

    train_model(args)