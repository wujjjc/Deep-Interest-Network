import torch
from torch import nn

class MLP(nn.Module):
    """多层感知机
    """
    def __init__(self, input_dim: int, hidden_dim: list, output_dim: int, activation=None, dropout=0.1):
        """_summary_

        Args:
            input_dim (_type_): 输入维度
            hidden_dim (_type_): 隐藏层维度
            output_dim (_type_): 输出维度
            activation (_type_): 激活函数
            dropout (float, optional): Dropout概率。
        """
        super(MLP, self).__init__()
        denses = []
        for dim in hidden_dim:
            denses.append(nn.Linear(input_dim, dim))
            if activation:
                denses.append(activation.__class__())
            denses.append(nn.Dropout(dropout))
            input_dim = dim
        denses.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*denses)
    def forward(self, x):
        return self.net(x)

class ActivationUnit(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=[80,40], dropout=0.1):
        """DIN的激活单元

        Args:
            embedding_dim (_type_): 输入的embedding维度
            hidden_dim (_type_): 隐藏层维度
            dropout (float, optional): Dropout概率。
        """
        super(ActivationUnit, self).__init__()
        self.mlp = MLP(embedding_dim * 4, hidden_dim, 1, activation=nn.PReLU(), dropout=dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, hist_item_emb, target_item_emb, neg=False, mask=None):
        # hist_item_emb: [batch_size, seq_len, embedding_dim]
        # target_item_emb: [batch_size, embedding_dim]
        # mask: [batch_size, seq_len]
        if not neg:
            target_item_emb = target_item_emb.unsqueeze(1).expand(hist_item_emb.size(0), hist_item_emb.size(1), target_item_emb.size(-1))# [batch_size, seq_len, embedding_dim]
            x = torch.cat([hist_item_emb, target_item_emb, hist_item_emb * target_item_emb, target_item_emb - hist_item_emb], dim=-1)
            x = (self.mlp(x) / (target_item_emb.shape[-1] ** 0.5)).squeeze(-1) # [batch_size, seq_len]
            # x = self.mlp(x).squeeze(-1) # [batch_size, seq_len]
            x = x.masked_fill(mask == 0, float('-inf')) # [batch_size, seq_len]
            return self.softmax(x) # [batch_size, seq_len]
        else:
        # hist_item_emb: [batch_size, seq_len, embedding_dim]
        # target_item_emb: [batch_size, neg_sample, embedding_dim]
            target_item_emb = target_item_emb.unsqueeze(2).expand(hist_item_emb.size(0), target_item_emb.size(1), hist_item_emb.size(1), target_item_emb.size(-1)) # [batch_size, neg_sample, seq_len, embedding_dim]
            hist_item_emb = hist_item_emb.unsqueeze(1).expand(hist_item_emb.size(0),  target_item_emb.size(1), target_item_emb.size(2), hist_item_emb.size(-1)) # [batch_size, neg_sample, seq_len, embedding_dim]
            x = torch.cat([hist_item_emb, target_item_emb, hist_item_emb * target_item_emb, target_item_emb - hist_item_emb], dim=-1) # [batch_size, neg_sample, seq_len, embedding_dim * 4]
            x = (self.mlp(x) / (target_item_emb.shape[-1] ** 0.5)).squeeze(-1) # [batch_size, neg_sample, seq_len]
            # x = self.mlp(x).squeeze(-1) # [batch_size, neg_sample, seq_len]
            x = x.masked_fill(mask.unsqueeze(1) == 0, float('-inf')) # [batch_size, neg_sample, seq_len]
            return self.softmax(x.squeeze(-1)) # [batch_size, neg_sample, seq_len]
            

class DIN(nn.Module):
    def __init__(self, num_items, num_users, num_categories, embedding_dim, hidden_dim):
        """_summary_

        Args:
            num_items (_type_): 物品数量
            num_users (_type_): 用户数量
            num_categories (_type_): 类别数量
            embedding_dim (_type_): embedding维度
            hidden_dim (_type_): MLP隐藏层维度
        """
        super(DIN, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        self.activation_unit = ActivationUnit(embedding_dim * 2)
        self.mlp = MLP(embedding_dim * 6, hidden_dim, 1, activation=nn.PReLU())
        self.user = MLP(embedding_dim * 2, [], embedding_dim * 2, activation=nn.PReLU())
        self.item_bias = nn.Embedding(num_items, 1, padding_idx=0)
    
    def forward(self, user_id, item_id, item_mask, item_category, item_category_length, target_item_id, target_category, target_category_length):
        """获得样本的预测结果

        Args:
            user_id (_type_):  用户id [batch_size, ]
            item_id (_type_): 历史物品id，shape: [batch_size, seq_len]
            item_mask (_type_): 历史物品序列掩码 [batch_size, seq_len]
            item_category (_type_): 历史物品类别id，shape: [batch_size, seq_len, category_len]
            item_category_length (_type_): 历史物品类别长度 [batch_size, seq_len]
            target_item_id (_type_): 目标物品id [batch_size, ]
            target_category (_type_): 目标物品类别id，shape: [batch_size, category_len]
            target_category_length (_type_): 目标物品类别长度 [batch_size, ]
        """
        # user_emb = self.user_embedding(user_id) # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_id) # [batch_size, seq_len, embedding_dim]
        target_item_emb = self.item_embedding(target_item_id) # [batch_size, embedding_dim]
        item_category_emb = self.category_embedding(item_category) # [batch_size, seq_len, category_len, embedding_dim]
        target_category_emb = self.category_embedding(target_category) # [batch_size, category_len, embedding_dim]
        # 对类别embedding进行平均池化
        item_category_emb = item_category_emb.sum(dim=2) / item_category_length.unsqueeze(-1).clamp(min=1) # [batch_size, seq_len, embedding_dim]
        target_category_emb = target_category_emb.sum(dim=1) / target_category_length.unsqueeze(-1).clamp(min=1) # [batch_size, embedding_dim]
        item_emb = torch.cat([item_emb, item_category_emb], dim=-1) # [batch_size, seq_len, embedding_dim * 2]
        target_item_emb = torch.cat([target_item_emb, target_category_emb], dim=-1) # [batch_size, embedding_dim * 2]
        attention = self.activation_unit(item_emb, target_item_emb, mask=item_mask) # [batch_size, seq_len]
        item_pool = torch.sum(attention.unsqueeze(-1) * item_emb, dim=1)  # [batch_size, embedding_dim * 2]
        user_emb = self.user(item_pool) # [batch_size, embedding_dim * 2]
        input = torch.cat([user_emb, target_item_emb, user_emb * target_item_emb], dim=-1) # [batch_size, embedding_dim * 6]
        output = self.mlp(input) + self.item_bias(target_item_id) # [batch_size, 1]
        if not self.training:
            output = torch.sigmoid(output) # [batch_size, 1]
        return output.squeeze(-1) # [batch_size, ]
    
    def neg_(self, user_id, item_id, item_mask, item_category, item_category_length, neg_item_id, neg_category, neg_category_length):
        """
            获取负样本的预测结果
        Args:
            user_id (_type_): 用户id [batch_size, ]
            item_id (_type_): 历史物品id，shape: [batch_size, seq_len]
            item_mask (_type_): 历史物品序列掩码 [batch_size, seq_len]
            item_category (_type_): 历史物品类别id，shape: [batch_size, seq_len, category_len]
            item_category_length (_type_): 历史物品类别长度 [batch_size, seq_len]
            neg_item_id (_type_): 负样本物品id，shape: [batch_size, neg_sample]
            neg_category (_type_): 负样本物品类别id，shape: [batch_size, neg_sample, category_len]
            neg_category_length (_type_): 负样本物品类别长度，shape: [batch_size, neg_sample]
        """
        # user_emb = self.user_embedding(user_id) # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_id) # [batch_size, seq_len, embedding_dim]
        neg_item_emb = self.item_embedding(neg_item_id) # [batch_size, neg_sample, embedding_dim]
        item_category_emb = self.category_embedding(item_category) # [batch_size, seq_len, category_len, embedding_dim]
        neg_category_emb = self.category_embedding(neg_category) # [batch_size, neg_sample, category_len, embedding_dim]
        # 对类别embedding进行平均池化
        item_category_emb = item_category_emb.sum(dim=2) / item_category_length.unsqueeze(-1).clamp(min=1) # [batch_size, seq_len, embedding_dim]
        neg_category_emb = neg_category_emb.sum(dim=2) / neg_category_length.unsqueeze(-1).clamp(min=1) # [batch_size, neg_sample, embedding_dim]
        item_emb = torch.cat([item_emb, item_category_emb], dim=-1) # [batch_size, seq_len, embedding_dim * 2]
        neg_item_emb = torch.cat([neg_item_emb, neg_category_emb], dim=-1) # [batch_size, neg_sample, embedding_dim * 2]
        attention = self.activation_unit(item_emb, neg_item_emb, mask=item_mask, neg=True) # [batch_size, neg_sample, seq_len]
        item_pool = torch.sum(attention.unsqueeze(-1) * item_emb.unsqueeze(1), dim=2)  # [batch_size, neg, embedding_dim * 2]
        user_emb = self.user(item_pool) # [batch_size, neg_sample, embedding_dim * 2]
        input = torch.cat([user_emb, neg_item_emb, user_emb * neg_item_emb], dim=-1) # [batch_size, neg_sample, embedding_dim * 6]
        output = self.mlp(input) + self.item_bias(neg_item_id) # [batch_size, neg_sample, 1]
        return output.squeeze(-1) # [batch_size, neg_sample]
    
    
        
        