import jsonlines
import torch
from config import Config
import ast
import random
"""
json
reviewerID	评论者的唯一标识符（ID），例如 "AO94DHGC771SJ"。
asin	Amazon Standard Identification Number，商品的唯一标识码（类似商品ID）。
reviewerName	评论者在亚马逊上显示的昵称（可能为空或匿名）。
helpful	一个列表 [有用票数, 总投票数]，表示其他用户认为该评论“有帮助”的投票情况。例如 [0, 0] 表示无人投票。
reviewText	评论文本的内容（用户写的长文本）。
overall	用户给出的综合评分，通常是 1.0 到 5.0 的浮点数（5 星最高）。
summary	评论的简短标题或总结（类似一句话摘要）。
unixReviewTime	评论创建时间的 Unix 时间戳（从 1970‑01‑01 开始的秒数），方便程序处理。
reviewTime	评论创建时间的可读格式，例如 "06 2, 2013"（月 日, 年）。

metadata
asin（Amazon Standard Identification Number）：亚马逊标准识别号，每个商品的唯一 ID，用于关联评论、图像等。
imUrl：商品图片的 URL 地址，可用来下载商品的主图。
description：商品的文字描述，通常是多行文本或列表，介绍商品的特点、规格等。
categories：商品所属的分类体系，通常是一个嵌套列表（树形结构），例如 [["Books", "Literature & Fiction", "Classics"]]。
title：商品的标题名称。
"""
def parse_line(line):
    line = line.strip()
    # 去掉外层双引号（如果存在）
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]
    # 将 Python 字面量（单引号字典）转换为真正的字典对象
    return ast.literal_eval(line)


def load_data(file_path):
    """
    读取json文件，再对reviewerID和asin和categories进行编码到整数，再把所有数据放入一个字典中，对字典中的数据按时间排序，最后返回这个字典
    reviewerID: 0-n
    asin: 1-m
    categories: 1-k
    :param file_path: json文件的路径
    :return:
    dict:
    {
        reviewerID:[(asin, overall, unixReviewTime), ............]  按照unixReviewTime排序
    }
    good_category: dict:{
        asin: [category_id, category_id, ...] 没有填充
    }
    num_items: item的总数
    num_users: user的总数
    num_categories: category的总数
    """
    print("正在读取数据...")
    with jsonlines.open(file_path) as reader:
        data = list(reader)
    review_map = {}
    asin_map = {}
    category_map = {}
    id_good_rating_time = {}
    good_data = {}
    good_category = {}
    with open(Config.data_path, 'r') as f:
        for raw_line in f:
            dic = parse_line(raw_line)
            # categories 是嵌套列表，如 [["Books", "Literature & Fiction", "Classics"]]，需要展平
            good_data[dic['asin']] = [c for path in dic["categories"] for c in path]
    for dic in data:
        reviewer_id = dic['reviewerID']
        asin = dic['asin']
        category = good_data.get(asin, [])
        for i in range(len(category)):
            if category[i] not in category_map:
                category_map[category[i]] = len(category_map) + 1
        if reviewer_id not in review_map:
            review_map[reviewer_id] = len(review_map)
        if asin not in asin_map:
            asin_map[asin] = len(asin_map) + 1
        if asin_map[asin] not in good_category:
            good_category[asin_map[asin]] = [category_map.get(c, 0) for c in category]
        id_good_rating_time.setdefault(review_map[reviewer_id], []).append((asin_map[asin], float(dic['overall']), int(dic['unixReviewTime'])))
    num_items = len(asin_map) + 1  # +1 因为 ID 从 1 开始，0 留给 padding
    num_users = len(review_map)
    num_categories = len(category_map) + 1  # 同理
    for reviewer_id in id_good_rating_time.keys():
        id_good_rating_time[reviewer_id].sort(key = lambda x: x[2])
    print("数据读取完成！")
    return id_good_rating_time, good_category, num_items, num_users, num_categories

def split_data(data, length = 100):
    """
    将数据划分为训练集和测试集
    :param data:dict:
    {
        reviewerID:[(asin, overall, unixReviewTime), ............]  按照unixReviewTime排序
    }
    :return:
    train_data: 训练集数据，不进行裁剪，格式 list [(reviewer_id, [(item_id, rating, time), ...], target), ...]
    test_data: 测试集数据，不进行裁剪，格式 list [(reviewer_id, [(item_id, rating, time), ...], target), ...]
    """
    train_data = []
    test_data = []
    for reviewer_id in data.keys():
        if len(data[reviewer_id]) < 2:
            continue
        for i in range(1, len(data[reviewer_id]) - 1):
            train_data.append((reviewer_id, data[reviewer_id][:i + 1]))
        test_data.append((reviewer_id, data[reviewer_id][:]))
    return train_data, test_data
class DinDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        reviewer_id, seq = self.data[idx]
        return reviewer_id, seq

def collate_fn(batch, good_category, num_items, neg_sample):
    """
    将一个batch的数据进行处理，返回一个字典，包含以下键值对：
    {
        'reviewer_id': reviewer_id,
        'item_id': item_id,
        'rating': rating,
        'target': target 正样本id,
        'mask': mask 历史序列掩码,
        'category': category,
        'category_length': category_length,
        'neg_sample': neg_sample 负样本id列表,
        'neg_category': neg_category 负样本类别列表,
        'neg_category_length': neg_category_length 负样本类别长度列表
    }
    :param batch: 一个batch的数据，格式为reviewer_id, [(item_id, rating, time), ...]
    :return: 处理后的数据字典
    """
    mx_len = 0
    reviewer_ids = []
    item_ids = []
    ratings = []
    targets = []
    masks = []
    categories = []
    category_lengths = []
    targets_category = []
    targets_category_length = []
    neg_samples = []
    neg_categories = []
    neg_category_lengths = []
    for reviewer_id, seq in batch:
        reviewer_ids.append(reviewer_id)
        target = seq[-1][0]  # 最后一个item的ID作为target
        reviews = seq[:-1]   # 除最后一个item外的历史序列
        item_id = [review[0] for review in reviews]
        rating = [review[1] for review in reviews]
        mx_len = max(mx_len, len(item_id))
        neg = []
        while len(neg) < neg_sample:
            neg_id = random.randint(1, num_items - 1)
            if neg_id not in item_id and neg_id != target and neg_id not in neg:
                neg.append(neg_id)
        length = len(reviews) - 1
        padding = Config.seq_len - length
        target_category_length = min(Config.max_category_len, len(good_category.get(target, [])))
        target_category = good_category.get(target, [])[-Config.max_category_len:] + [0] * (Config.max_category_len - target_category_length)
        neg_category_length = [min(Config.max_category_len, len(good_category.get(neg_id, []))) for neg_id in neg]
        neg_category = [good_category.get(neg_id, [])[-Config.max_category_len:] + [0] * (Config.max_category_len - neg_category_length[i]) for i, neg_id in enumerate(neg)]
        item_ids.append(item_id)
        ratings.append(rating)
        targets.append(target)
        targets_category.append(target_category)
        targets_category_length.append(target_category_length)
        neg_samples.append(neg)
        neg_categories.append(neg_category)
        neg_category_lengths.append(neg_category_length)
    for i in range(len(item_ids)):
        padding = mx_len - len(item_ids[i])
        mask = [0] * padding + [1] * len(item_ids[i])
        item_ids[i] = [0] * padding + item_ids[i]
        ratings[i] = [0] * padding + ratings[i]
        category_length = [min(Config.max_category_len, len(good_category.get(item, []))) for item in item_ids[i]]
        category = [good_category.get(item_ids[i][j], [])[-Config.max_category_len:] + [0] * (Config.max_category_len - category_length[j])
                    for j in range(len(item_ids[i]))]
        categories.append(category)
        category_lengths.append(category_length)
        masks.append(mask)
    reviewer_ids = torch.tensor(reviewer_ids, dtype=torch.long) # [B, ]
    item_ids = torch.tensor(item_ids, dtype=torch.long) # [B, seq_len]
    ratings = torch.tensor(ratings, dtype=torch.float) # [B, seq_len]
    targets = torch.tensor(targets, dtype=torch.long) # [B, ]
    masks = torch.tensor(masks, dtype=torch.float) # [B, seq_len]
    categories = torch.tensor(categories, dtype=torch.long) # [B, seq_len, category_len]
    category_lengths = torch.tensor(category_lengths, dtype=torch.long) # [B, seq_len]
    targets_category = torch.tensor(targets_category, dtype=torch.long) # [B, category_len]
    targets_category_length = torch.tensor(targets_category_length, dtype=torch.long) # [B, ]
    neg_samples = torch.tensor(neg_samples, dtype=torch.long) # [B, neg_sample]
    neg_categories = torch.tensor(neg_categories, dtype=torch.long) # [B, neg_sample, category_len]
    neg_category_lengths = torch.tensor(neg_category_lengths, dtype=torch.long) # [B, neg_sample]
    return {
        'reviewer_id': reviewer_ids,
        'item_id': item_ids,
        'rating': ratings,
        'target': targets,
        'mask': masks,
        'category': categories,
        'category_length': category_lengths,
        'target_category': targets_category,
        'target_category_length': targets_category_length,
        'neg_sample': neg_samples,
        'neg_category': neg_categories,
        'neg_category_length': neg_category_lengths
    }