import torch 
from torch import fbgemm_linear_fp16_weight, nn 
import torch.nn.functional as F 

class MyCEL(torch.nn.Module):
    def __init__(self):
        super(MyCEL, self).__init__()

    def forward(self,policy, target_policy):
  
       
        total_error = torch.sum(-target_policy*(1e-8+policy).log(),1).mean()
        return total_error



class ConvBlock(nn.Module):
    def __init__(self,inplanes=9,outplanes=256):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3, stride=1, padding=1)
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

class ResTower(nn.Module):
    def __init__(self,resblocks,conv_in_planes=9,conv_out_planes=256,res_out_planes = 256):
        super(ResTower,self).__init__()
        self.res_num = resblocks
        self.conv = ConvBlock(conv_in_planes,conv_out_planes)
        for block in range(self.res_num):
            setattr(self,f'res_{block}',ResBlock(inplanes=conv_out_planes,planes=res_out_planes))
        self.conv2 = ConvBlock(res_out_planes,1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,s):
        s = self.conv(s)
        for block in range(self.res_num):
            s = getattr(self,f'res_{block}')(s)
        s = self.conv2(s)
        s = nn.Softmax(2)(s.view(1, 1, -1)).view_as(s)
        return s




class PolicyNet(nn.Module):
    def __init__(self,base,head = ResTower(2,conv_in_planes=1)):
        super(PolicyNet,self).__init__()
        self.base = base
        self.head = head
        self.freezeBase()
    
    def freezeBase(self):
        for param in self.base.parameters():
            param.requires_grad = False
    
    def unfreezeBase(self):
        for param in self.base.parameters():
            param.requires_grad = True
    
    def forward(self,s):
        s = self.base(s)
        start_sq = s
        end_sq = self.head(s)
        return end_sq.view(-1,64)


class PolicyNet2(nn.Module):
    def __init__(self,base,head = ResTower(resblocks = 2, conv_in_planes=10),conv_in_planes = 9,conv_out_planes=256,res_out_planes =256):
        super(PolicyNet2,self).__init__()
        self.base = base
        self.head = head 
        self.freezeBase()


    def freezeBase(self):
        for param in self.base.parameters():
            param.requires_grad = False
    
    def unfreezeBase(self):
        for param in self.base.parameters():
            param.requires_grad = True
    

    def forward(self,s):
        residual = s
        s = self.base(s)

        s = torch.cat((s,residual),dim = 1)
        ## s = s + residual 
        end_sq = self.head(s)
        return end_sq.view(-1,64)


class ValueNet(nn.Module):
    def __init__(self,resblocks,inplanes = 9,outplanes = 256):
        super(ValueNet,self).__init__()
        self.conv = ConvBlock(inplanes = inplanes , outplanes = outplanes)
        self.restower = ResTower(resblocks, conv_in_planes= outplanes, conv_out_planes=outplanes,res_out_planes=outplanes)
        self.fc1 = nn.Linear(8*8,1)
    def forward(self,s):
        s = self.conv(s)
        s = self.restower(s)
        s = self.fc1(s.view(-1,8*8))
        s = torch.tanh(s)
        return s 
    

class ValueNet2(nn.Module):
    def __init__(self,resblocks, inplanes = 9,outplanes = 256):
        super(ValueNet2,self).__init__()
        self.restower1 = ResTower(resblocks, conv_in_planes=inplanes , conv_out_planes= outplanes,res_out_planes=outplanes)
        self.restower2 = ResTower(resblocks, conv_in_planes= (inplanes+1) , conv_out_planes= outplanes,res_out_planes= outplanes)
        self.fc = nn.Linear(8*8,1)
    
    def forward(self,s):
        residual = s
        s = self.restower1(s)
        s = torch.cat((s,residual),dim = 1)
        s = self.restower2(s)
        print(s.shape)
        s = self.fc(s.view(-1,8*8))


        return torch.tanh(s)





        

    



    




if __name__ == '__main__':
    t = torch.rand(10,9,8,8)
#     base = ResTower(2)
#     head = ResTower(resblocks = 2, conv_in_planes=1)
#     net2 = PolicyNet2(base)
#     s,out = net2(t)
#     print(s.shape,out.shape)
# #     net = PolicyNet(base,head)
# #     s , e = net(t)
# #     print(s.shape, e.shape)
# #     #p = p.squeeze(0).squeeze(0)
# #     print(s)
# #     print(torch.sum(s))
# #     print('####################################')
# #     print(e)
# #     print(torch.sum(e))
#     # criterion = MyCEL()
#     # y = torch.rand(1,1,8,8)
#     # loss = criterion(s.view(1,-1),y.view(1,-1))
#     # print('loss:', loss.item())
    policy = PolicyNet(ResTower(2))
    policy2 = PolicyNet(ResTower(2))
    policy.eval()
    policy2.eval()
    p1 = policy(t)
    p2 = policy2(t)
    print(p1.shape,p2.shape,policy2.base(t).shape)

        
