import torch
import torch.nn as nn


### 动态图卷积层
class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()
        # 生成静态图的邻接矩阵
        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        # 对输入特征进行线性变换，得到新的特征表示
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))
        # 将特征图的空间维度（N）压缩成一个标量，以便后续进行全局处理。
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 对全局特征进行卷积操作
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        # 对全局特征进行批归一化。
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)
        # 用于从特征图中生成动态图的邻接矩阵。
        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        # 对动态图卷积后的特征进行线性变换
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    # 对输入进行静态图卷积操作
    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    # 计算全局特征，生成动态邻接矩阵
    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    # 使用动态邻接矩阵进行卷积操作
    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    # 将静态图卷积和动态图卷积组合起来，进行前向传播
    def forward(self, x):
        """ D-GCN module
        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return dynamic_adj, x


class ADD_GCN(nn.Module):
    def __init__(self, model, num_classes, hash_code_length=64):
        super(ADD_GCN, self).__init__()
        # 包含了 ResNet-50 的卷积层和残差块，用于特征提取
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        # 将特征图转化为类别预测的权重
        self.fc = nn.Conv2d(model.fc.in_features, num_classes, (1,1), bias=False)
        # 对 ResNet 输出的特征图进行卷积操作，将特征维度从 2048 压缩到 1024
        self.conv_transform = nn.Conv2d(2048, 1024, (1,1))
        self.relu = nn.LeakyReLU(0.2)
        # 实例化 DynamicGraphConvolution 模块，用于图卷积操作。
        self.gcn = DynamicGraphConvolution(1024, 1024, num_classes)
        
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.last_linear = nn.Conv1d(1024, self.num_classes, 1)

        ### 哈希层
        self.hash_layer = nn.Linear(20, hash_code_length, bias=False)
        self.hash_code_length = hash_code_length
        self.hash_sign = nn.Tanh()
        self.hash_layer.weight.data.normal_(0,0.01)
        # image normalization
        
    # 使用 ResNet 的特征提取层进行特征提取
    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward_classification_sm(self, x):
        """ Get another confident scores {s_m}.
        Shape:
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out) # C_out: num_classes
        """
        x = self.fc(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = x.topk(1, dim=-1)[0].mean(dim=-1)
        return x

    # 通过生成的掩码对特征进行加权。
    def forward_sam(self, x):
        """ SAM module
        Shape: 
        - Input: (B, C_in, H, W) # C_in: 2048
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        mask = self.fc(x) 
        mask = mask.view(mask.size(0), mask.size(1), -1) 
        mask = torch.sigmoid(mask)
        mask = mask.transpose(1, 2)

        x = self.conv_transform(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.matmul(x, mask)
        return x

    def forward_dgcn(self, x):
        x = self.gcn(x)
        return x

    # 进行特征提取、分类、SAM 和图卷积操作，然后结合结果输出最终预测
    def forward(self, x):

        x = self.forward_feature(x)

        out1 = self.forward_classification_sm(x)

        v = self.forward_sam(x) # B*1024*num_classes
        dag, dag_ = self.forward_dgcn(v) # B*num_classes*num_classes
        z = v + dag_

        out2 = self.last_linear(z) # B*1*num_classes
        mask_mat = self.mask_mat.detach()
        out2 = (out2 * mask_mat).sum(-1)
        
        # 生成哈希码
        hash_code = self.hash_layer(v)  # 生成哈希码
        hash_code = self.hash_sign(hash_code)
        
        return (out1 + out2) / 2, dag, hash_code


    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

model_dict = {'ADD_GCN': ADD_GCN}
import torchvision

def get_model(num_classes):
    res50 = torchvision.models.resnet50(pretrained=True)
    model = model_dict['ADD_GCN'](res50, num_classes)
    return model
