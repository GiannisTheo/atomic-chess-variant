import chess
import chess.variant
import chess.pgn
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from encoder import encode
from move_encoder import decode_move, mask_invalid,get_best_move


class ConvBlock(nn.Module):
    def __init__(self,outplanes=256):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(9, outplanes, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(outplanes)

    def forward(self, s):
        s = self.conv1(s)
        s= self.bn1(s)
        s=F.relu(s)
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class PolicyOut(nn.Module):
    def __init__(self):
        super(PolicyOut, self).__init__()
        self.conv=nn.Conv2d(256,128,kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        self.fc=nn.Linear(8*8*128,128*32)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.conv(x)
        x=F.relu(self.bn(x))
        x=x.view(-1,8*8*128)
        x=self.softmax(self.fc(x)).exp()
        return x

class ValueOut(nn.Module):
    def __init__(self):
        super(ValueOut, self).__init__()
        self.conv=nn.Conv2d(256,128,kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        self.fc1=nn.Linear(8*8*128,128*8)
        self.fc2=nn.Linear(128*8,512)
        self.fc3=nn.Linear(512,1)

    def forward(self,x):
        x=self.conv(x)
        x=F.relu(self.bn(x))
        x=x.view(-1,8*8*128)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=torch.tanh(self.fc3(x))
        return x
    
class PolicyNet(nn.Module):
    def __init__(self,res_num):
        super(PolicyNet,self).__init__()
        self.res_num=res_num
        self.conv=ConvBlock()
        for block in range(self.res_num):
              setattr(self, "res_%i" % block,ResBlock())
        self.out=PolicyOut()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(self.res_num):
            s = getattr(self, "res_%i" % block)(s)
        s=self.out(s)
        return s

class ValueNet(nn.Module):
    def __init__(self,res_num):
        super(ValueNet,self).__init__()
        self.res_num=res_num
        self.conv=ConvBlock()
        for block in range(self.res_num):
              setattr(self, "res_%i" % block,ResBlock())
        self.out=ValueOut()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(self.res_num):
            s = getattr(self, "res_%i" % block)(s)
        s=self.out(s)
        return s



if __name__=="__main__":
    board=chess.variant.AtomicBoard()
    state=torch.Tensor(encode(board.fen())).unsqueeze(0)
    pmodel=PolicyNet(1)
    vmodel=ValueNet(1)
    p=pmodel(state)
    v=vmodel(state)
    print("value=",v.item())
    p=p.squeeze(0).detach().cpu().numpy()
    p=mask_invalid(p,board)
    print(get_best_move(p,board))
    #block = ConvBlock(outplanes=256)
    #resblock = ResBlock()
    #outblock = PolicyOut()
    #out= block.forward(state)
    #print(out.shape)
    #out2=resblock.forward(out)
    #print(out2.shape)
    #out3=outblock.forward(out2)
    #print(out3.shape)
    #p=out3.squeeze(0).detach().cpu().numpy()
    #p=mask_invalid(p,board)
    #move=get_best_move(p,board)
    #print(move)
    