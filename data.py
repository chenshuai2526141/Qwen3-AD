from modelscope.msdatasets import MsDataset
import json
import random
import os
import time
from pathlib import Path


def _convert_advertisegen_cache_gbk_to_utf8() -> None:
    """AdvertiseGen 官方 CSV 为 GBK，ModelScope 用默认 UTF-8 读会失败；将缓存目录内非 UTF-8 文件转为 UTF-8。"""
    root = (
        Path.home()
        / ".cache"
        / "modelscope"
        / "hub"
        / "datasets"
        / "lvjianjin"
        / "AdvertiseGen"
        / "master"
        / "data_files"
    )
    if not root.is_dir():
        return
    for f in root.iterdir():
        if not f.is_file():
            continue
        raw = f.read_bytes()
        if not raw:
            continue
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("gbk")
            except UnicodeDecodeError:
                continue
            f.write_text(text, encoding="utf-8")
            print(f"已将 GBK 数据文件转为 UTF-8: {f.name}")


# 设置ModelScope镜像源
os.environ['MODELSCOPE_ENDPOINT'] = 'https://modelscope.oss-cn-beijing.aliyuncs.com'

# 设置随机种子以确保可重复性
random.seed(42)

# 加载数据集，添加重试机制
max_retries = 3
for attempt in range(max_retries):
    try:
        _convert_advertisegen_cache_gbk_to_utf8()
        print(f"尝试加载数据集 (第 {attempt + 1} 次)...")
        ds = MsDataset.load('lvjianjin/AdvertiseGen', subset_name='default', split='train')
        print("数据集加载成功！")
        break
    except UnicodeDecodeError:
        _convert_advertisegen_cache_gbk_to_utf8()
        print(f"第 {attempt + 1} 次尝试失败: 编码问题，已尝试修复缓存后重试")
        if attempt < max_retries - 1:
            time.sleep(1)
        else:
            raise
    except Exception as e:
        print(f"第 {attempt + 1} 次尝试失败: {str(e)}")
        if attempt < max_retries - 1:
            print("等待5秒后重试...")
            time.sleep(5)
        else:
            print("所有重试都失败了，请检查网络连接或数据集是否存在")
            raise e

# 将数据集转换为列表
data_list = list(ds)
full_count = len(data_list)

# 随机打乱后仅保留约全量的 1/7（向下取整，至少 1 条）
random.shuffle(data_list)
keep_count = max(1, full_count // 7)
data_list = data_list[:keep_count]
print(f"全量样本数: {full_count}，截取 1/7 后用于划分: {keep_count}")

# 计算分割点（在子集上仍按 9:1 划分）
split_idx = int(len(data_list) * 0.9)

# 分割数据
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

# 保存训练集
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# 保存验证集
with open('val.jsonl', 'w', encoding='utf-8') as f:
    for item in val_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print("数据集已分割完成（子集内约 9:1 train/val）：")
print(f"训练集大小：{len(train_data)}")
print(f"验证集大小：{len(val_data)}")