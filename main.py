import torch
from config import Config
from data import *
from model import DIN
from train import train
from gpu import *
import os
device = select_gpu() 
id_good_rating_time, good_category, num_items, num_users, num_categories = load_data(Config.json_path)
train_data, test_data = split_data(id_good_rating_time, Config.seq_len)
train_dataset = DinDataset(train_data)
test_dataset = DinDataset(test_data)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, good_category, num_items, 1), num_workers=4, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, good_category, num_items, 1), num_workers=4, pin_memory=True)
net = DIN(num_items, num_users, num_categories, Config.embedding_dim, [80, 40]).to(device)
# if os.path.exists('best.pth'):
#     print("加载模型...")
#     net.load_state_dict(torch.load('best.pth', map_location=device))
optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
criterion = torch.nn.BCEWithLogitsLoss()
train(net, train_dataloader, test_dataloader, optimizer, scheduler, criterion, device)
