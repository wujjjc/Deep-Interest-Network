import subprocess
import torch


def get_nvidia_smi_to_cuda_map():
    """建立 nvidia-smi 编号到 CUDA 编号的映射

    nvidia-smi 使用 PCI 总线顺序编号，PyTorch CUDA 使用 CUDA_VISIBLE_DEVICES 顺序，
    两者可能不一致。通过 uuid 建立映射。

    Returns:
        dict: {nvidia_smi_id: cuda_id}
    """
    # 获取 nvidia-smi 中的 uuid 和编号
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    smi_uuid = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.split(', ')
        smi_id, uuid = int(parts[0].strip()), parts[1].strip()
        # nvidia-smi 返回的 uuid 带 "GPU-" 前缀，PyTorch 不带，统一去掉
        smi_uuid[smi_id] = uuid.replace('GPU-', '')

    # 获取 CUDA 设备的 uuid
    cuda_uuid = {}
    for cuda_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(cuda_id)
        cuda_uuid[cuda_id] = str(props.uuid)  # UUID 字符串

    # 建立映射：nvidia_smi_id -> cuda_id
    uuid_to_cuda = {v: k for k, v in cuda_uuid.items()}
    mapping = {}
    for smi_id, uuid in smi_uuid.items():
        if uuid in uuid_to_cuda:
            mapping[smi_id] = uuid_to_cuda[uuid]

    return mapping


def select_gpu(min_free_memory_gb=0):
    """选择最空闲的 GPU，优先显存占用最少

    Args:
        min_free_memory_gb: 最低可用显存要求(GB)，低于此值的 GPU 不考虑

    Returns:
        torch.device: 选中的 GPU 设备
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return torch.device('cuda:0')

    # 获取 nvidia-smi 到 cuda 的映射
    smi_to_cuda = get_nvidia_smi_to_cuda_map()

    # 通过 nvidia-smi 获取每张卡的显存使用情况
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )

    best_gpu = None
    best_free = -1

    for line in result.stdout.strip().split('\n'):
        parts = line.split(', ')
        smi_id = int(parts[0].strip())
        mem_used = float(parts[1].strip())  # MB
        mem_total = float(parts[2].strip())  # MB

        if smi_id not in smi_to_cuda:
            continue

        cuda_id = smi_to_cuda[smi_id]
        free_memory = mem_total - mem_used  # MB

        # 检查是否满足最低显存要求
        if free_memory < min_free_memory_gb * 1024:
            continue

        # 选择显存最空闲的 GPU
        if free_memory > best_free:
            best_free = free_memory
            best_gpu = cuda_id

    if best_gpu is None:
        print("警告：没有满足条件的 GPU，使用 cuda:0")
        return torch.device('cuda:0')

    print(f"选择 GPU cuda:{best_gpu}（nvidia-smi:{[k for k,v in smi_to_cuda.items() if v == best_gpu][0]}），"
          f"可用显存: {best_free / 1024:.1f} GB")
    return torch.device(f'cuda:{best_gpu}')


if __name__ == '__main__':
    device = select_gpu()
    print(f"使用设备: {device}")
