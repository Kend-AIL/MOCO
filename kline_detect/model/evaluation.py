import torch

def evaluate(predictions, targets, threshold=0.5):
    """
    Evaluate the model predictions.

    Args:
    predictions: torch.Tensor, model predictions (batch_size, 1536)
    targets: torch.Tensor, target values (batch_size, 1536)
    threshold: float, threshold for binary classification (default is 0.5)

    Returns:
    accuracy: float, accuracy of the model
    precision: float, precision of the model
    recall: float, recall of the model
    f1_score: float, F1 score of the model
    """

    # 将概率转换为二进制预测
    binary_predictions = (predictions > threshold).int()

    # 计算 true positives、false positives、true negatives、false negatives
    tp = torch.sum((binary_predictions == 1) & (targets == 1))
    fp = torch.sum((binary_predictions == 1) & (targets == 0))
    tn = torch.sum((binary_predictions == 0) & (targets == 0))
    fn = torch.sum((binary_predictions == 0) & (targets == 1))

    # 计算评估指标
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return accuracy.item(), precision.item(), recall.item(), f1_score.item()