import torch
import numpy as np
from openke.module.model import TransE, RotatE
from openke.data import TrainDataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

def load_model(in_path, ckpt_path, model_name="transe"):
    # 1. 加载和训练时一样的 DataLoader (只为获取 ent_tot 和 rel_tot 结构)
    train_dataloader = TrainDataLoader(
        in_path=in_path, 
        nbatches=100,
        threads=8, 
        sampling_mode="normal", 
        bern_flag=1, 
        filter_flag=1, 
        neg_ent=25,
        neg_rel=0
    )

    # 2. 初始化结构一致的模型
    if model_name.lower() == "transe":
        model = TransE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=200, 
            p_norm=1, 
            norm_flag=True
        )
    elif model_name.lower() == "rotate":
        model = RotatE(
            ent_tot=train_dataloader.get_ent_tot(),
            rel_tot=train_dataloader.get_rel_tot(),
            dim=200, 
            margin=6.0,
            epsilon=2.0
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 3. 加载已经训练好的权重
    model.load_checkpoint(ckpt_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        
    return model, train_dataloader.get_rel_tot()

def multi_class_pr_auc(y_true, y_score, num_classes):
    """计算多分类的 PR AUC (Macro Average)"""
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    pr_aucs = []
    # label_binarize 对于二分类可能会返回一维，但是按我们需要的是0,1,2所以固定3类安全
    for i in range(num_classes):
        pr_aucs.append(average_precision_score(y_true_bin[:, i], y_score[:, i]))
    return np.mean(pr_aucs)

def get_relation_probabilities(scores, threshold):
    """
    将横向距离得分转换为概率分布 (近似Softmax或自定义映射)。
    距离越小，概率越大。
    Label 0 的概率 = 距离超出 threshold 的信心。
    """
    # 将距离转为相似度 (例如 e^-score或使用距离翻转)
    # OpenKE里的打分一般不会变为绝对Softmax。这里我们需要构造成三分类概率 [prob(0), prob(1), prob(2)]
    # score[0] 为 relation=0 的距离，score[1] 为 relation=1, score[2] 为 relation=2
    # 注意：我们的模型实际上只训练了 relation 1 和 2，如果在测试集遇到了 relation 0怎么办？
    # 因为 TransE 不原生支持 0 的 "非结构”，我们把 (threshold - min(scores)) 转化成一个响应值
    # 如果 min(scores) 远大于 threshold，给 0 类极大权重。
    
    # 简单启发式映射，用来计算 AUC 和 PRC 
    # 对于每个样本，构造出一个长度为 3 的 P 向量
    probs = np.zeros((scores.shape[0], 3))
    
    # 假设关系ID 1 和 2 就是原来的索引
    for i in range(scores.shape[0]):
        dist_1 = scores[i, 1] if 1 < scores.shape[1] else 999 
        dist_2 = scores[i, 2] if 2 < scores.shape[1] else 999
        
        # 将距离映射成某种逻辑概率的 score (分越大，概率越高)
        # 用 max_dist 减去 当前距离 当作分数，再做 soft归一化
        # Label 0 的分数：阈值 - 真实关系的最小距离 （距离越大 => 得分越小 => 0的置信度越大）
        min_dist = min(dist_1, dist_2)
        score_0 = dist_1 + dist_2 # 假装它离得很远，或者是直接：
        
        s_1 = max(threshold - dist_1, 0)
        s_2 = max(threshold - dist_2, 0)
        
        if min_dist > threshold:
            # 肯定是 0
            s_0 = min_dist - threshold  
            s_1, s_2 = 0, 0
        else:
            s_0 = 0
            
        total = s_0 + s_1 + s_2 + 1e-8
        probs[i] = [s_0/total, s_1/total, s_2/total]
    
    return probs

def evaluate_on_test_set(test_file, model, rel_tot, threshold=5.0, batch_size=1024):
    """
    读取 test2id.txt 进行批量预测并计算准确率。
    """
    print(f"Reading test data from {test_file}...")
    with open(test_file, 'r') as f:
        lines = f.readlines()
        
    num_triples = int(lines[0].strip())
    test_triples = lines[1:num_triples+1]
    
    true_labels = []
    heads = []
    tails = []
    
    for line in test_triples:
        h, t, r = line.strip().split()
        heads.append(int(h))
        tails.append(int(t))
        true_labels.append(int(r))
        
    true_labels = np.array(true_labels)
    preds = []
    all_scores_list = []
    
    print("Running predictions in batches...")
    
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(heads), batch_size)):
            batch_h = heads[i:i+batch_size]
            batch_t = tails[i:i+batch_size]
            
            # 每个样本扩张 rel_tot 次
            # 例: bs=2, rel_tot=3
            # heads = [h1, h1, h1, h2, h2, h2]
            expanded_h = np.repeat(batch_h, rel_tot)
            expanded_t = np.repeat(batch_t, rel_tot)
            expanded_r = np.tile(np.arange(rel_tot), len(batch_h))
            
            h_tensor = torch.tensor(expanded_h)
            t_tensor = torch.tensor(expanded_t)
            r_tensor = torch.tensor(expanded_r)
            
            if torch.cuda.is_available():
                h_tensor = h_tensor.cuda()
                t_tensor = t_tensor.cuda()
                r_tensor = r_tensor.cuda()
            
            # 必须传 'mode': 'normal' 告诉底层是正常前向而不是 'head_batch' / 'tail_batch' (链接预测)
            data = {'batch_h': h_tensor, 'batch_t': t_tensor, 'batch_r': r_tensor, 'mode': 'normal'}
            scores = model.predict(data)
            
            # 在某些版本的 OpenKE 中，predict() 返回的已经是 numpy ndarray，因此不需要再 .cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            # scores 是 1D array，长度为 batch_size * rel_tot
            # 我们 reshape 成 (batch_size, rel_tot)
            scores = scores.reshape(-1, rel_tot)
            all_scores_list.append(scores)
            
            # 按照阈值判断
            for row in scores:
                # 寻找除了关系 0（假设本来不是关系而是无边）以外能够匹配的最佳真实关系距离
                # 如果只有关系 1 和 2 是训练的，那么这里我们可以忽略 `0` 号关系产生的打分
                best_rel = np.argmin(row)
                min_score = row[best_rel]
                
                # if min_score > threshold:
                #     preds.append(0)
                # else:
                preds.append(best_rel)
                    
    preds = np.array(preds)
    all_scores_matrix = np.vstack(all_scores_list)
    
    # 构造概率，用来计算 AUC 和 PR AUC
    probs = get_relation_probabilities(all_scores_matrix, threshold)
    
    print("\nCalculating Metrics...")
    
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="macro")
    
    # auc和prc要求概率
    try:
        auc = roc_auc_score(true_labels, probs, multi_class="ovr")
        prc = multi_class_pr_auc(true_labels, probs, num_classes=3)
    except Exception as e:
        print(f"AUC/PRC calculation failed (may be due to missing classes in test): {e}")
        auc, prc = 0.0, 0.0

    pre = precision_score(true_labels, preds, average="macro", zero_division=0)
    rec = recall_score(true_labels, preds, average="macro", zero_division=0)
    
    # [关键修改] 获取每个类别的详细 Recall
    rec_per_class = recall_score(true_labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    pre_per_class = precision_score(true_labels, preds, average=None, labels=[0, 1, 2], zero_division=0)
    
    # 提取 Label 2 (Negative) 的指标
    rec_label2 = rec_per_class[2] if len(rec_per_class) > 2 else 0.0
    pre_label2 = pre_per_class[2] if len(pre_per_class) > 2 else 0.0

    print("=" * 40)
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {f1:.4f}")
    print(f"AUC (OVR): {auc:.4f}")
    print(f"PR AUC   : {prc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("-" * 40)
    print(f"Class 0 (No Edge) - Prec: {pre_per_class[0]:.4f}, Rec: {rec_per_class[0]:.4f}")
    print(f"Class 1 (Pos Edge)- Prec: {pre_per_class[1]:.4f}, Rec: {rec_per_class[1]:.4f}")
    print(f"Class 2 (Neg Edge)- Prec: {pre_label2:.4f}, Rec: {rec_label2:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    in_path = "/mnt/e/OpenKE/benchmarks/PPI/"
    ckpt_path = "/mnt/e/OpenKE/checkpoint/transe_PPI_UPDATE.ckpt"
    test_file = "/mnt/e/OpenKE/benchmarks/PPI/valid2id.txt"  # 如果你没有存 test2id，你可以用 valid2id 测试
    
    # transe or rotate
    model_name = "transe" 
    
    print(f"Loading {model_name} model...")
    model, rel_tot = load_model(in_path, ckpt_path, model_name=model_name)
    print("Model loaded successfully.\n")

    # 注意：threshold 这里设置了 5.0（因为你的 marginloss.margin 是 5.0，超过该值大概率是假边）
    # 你可以手动调整阈值以观测评价指标变化
    evaluate_on_test_set(test_file, model, rel_tot, threshold=5.0)
