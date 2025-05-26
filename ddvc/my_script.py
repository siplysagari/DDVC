import torch
import numpy as np
def calculate_similarity(tensor_matrix):
    similarities = []
    for i in range(1, tensor_matrix.size(0)):  # 从第一个元素到最后一个
        # 计算 tensor_matrix[i] 与 tensor_matrix[i-1] 的余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(tensor_matrix[i], tensor_matrix[i-1], dim=0)
        similarities.append(cos_sim.item())
    return similarities

def find_intervals(lst,std_factor=1):
    intervals = []
    start=-1
    mean_sim = np.mean(lst)
    std_sim = np.std(lst)
    
    # 设置自适应阈值 (可以调整 std_factor 来控制分段的灵敏度)
    threshold = mean_sim - std_factor * std_sim
    for i in range(0, len(lst)):
        # if lst[i]>0.87:
        if lst[i]>threshold:
            if start==-1:
                start=i
            continue
        else:
            if start<i and start!=-1:
                if start!= i-1:
                    intervals.append([start, i])
                start=-1

    # Append the last interval if it hasn't been added
    intervals.append([start, len(lst)])
    return intervals


def new_find_intervals(lst,std_factor=1):
    intervals = []
    start=0
    mean_sim = np.mean(lst)
    std_sim = np.std(lst)
    
    # 设置自适应阈值 (可以调整 std_factor 来控制分段的灵敏度)
    threshold = mean_sim - std_factor * std_sim
    # print(threshold)
    for i in range(len(lst)):
        # if lst[i]>0.87:
        if lst[i]>=threshold:
            # if start==-1:
            #     start=i
            continue
        else:
            if start<i:
                intervals.append([start, i])
                start=i+1

    # Append the last interval if it hasn't been added
    intervals.append([start, len(lst)])
    return intervals


def merge_tensor_matrix(tensor_matrix, merge_ranges):
    # 初始化一个空列表，用于存储合并后的张量
    merged_tensors = []
    
    for a, b in merge_ranges:
        # 提取从 a 到 b 的所有张量
        tensors_to_merge = tensor_matrix[a:b+1]  # b+1 因为索引是半开区间
        
        # 对这些张量取均值
        merged_tensor = tensors_to_merge.mean(dim=0)
        
        # 将均值化的张量存入列表
        merged_tensors.append(merged_tensor)
    
    return torch.stack(merged_tensors)  # 将结果拼接成新的张量矩阵

def segment_by_similarity(similarity_list, std_factor=1.0):
    # 计算相似度的均值和标准差
    mean_sim = np.mean(similarity_list)
    std_sim = np.std(similarity_list)
    
    # 设置自适应阈值 (可以调整 std_factor 来控制分段的灵敏度)
    threshold = mean_sim - std_factor * std_sim
    
    # 初始化段落的开始点
    segments = []
    start = 0
    
    for i in range(len(similarity_list)):
        if similarity_list[i] < threshold:
            # 当相似度低于阈值，认为是分段点
            segments.append((start, i + 1))  # [start, i+1) 为一个段落
            start = i + 1
    
    # 最后一段
    if start < len(similarity_list):
        segments.append((start, len(similarity_list)))
    
    return segments


def feature_project(source_feature, target_feature):
    temp_tensor = torch.randn_like(source_feature)
    source_feature /= source_feature.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        sim = source_feature @ target_feature.T.float()
        sim = (sim * args.tp).softmax(dim=-1)
        prefix_embedding = sim @ target_feature.float()
        prefix_embedding /= prefix_embedding.norm(dim=-1, keepdim=True)
        temp_tensor = prefix_embedding.detach()
    return temp_tensor

def multi_scale_aggregation(input_tensor, n=2, levels=4):
    """
    Perform multi-scale aggregation on the input tensor, considering padding influence.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (1, T, C).
        n (int): Step size for each aggregation.
        levels (int): Number of aggregation levels.

    Returns:
        List[torch.Tensor]: List of tensors, each representing an aggregation level.
    """
    if input_tensor.dim() != 3 or input_tensor.size(0) != 1:
        raise ValueError("Input tensor must have shape (1, T, C).")

    aggregated = input_tensor
    input_conv_list = []

    for level in range(levels):
        T, C = aggregated.shape[1], aggregated.shape[2]
        # if T < n:
        #     # Stop aggregation if temporal dimension is too small
        #     break

        # Calculate the number of chunks based on step size `n`
        num_chunks = (T + n - 1) // n

        # Create a mask to indicate valid positions
        mask = torch.ones(1, T, 1, device=input_tensor.device)
        pad_size = num_chunks * n - T
        if pad_size > 0:
            aggregated = torch.nn.functional.pad(aggregated, (0, 0, 0, pad_size), mode='constant', value=0)
            mask = torch.nn.functional.pad(mask, (0, 0, 0, pad_size), mode='constant', value=0)

        # Reshape and compute mean pooling with mask
        aggregated = aggregated.view(1, num_chunks, n, C)
        mask = mask.view(1, num_chunks, n, 1)

        # Avoid division by zero
        weighted_mean = (aggregated * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1e-6)
        # print(mask)
        input_conv_list.append(weighted_mean)

        # Update aggregated for the next level
        aggregated = weighted_mean

    return input_conv_list