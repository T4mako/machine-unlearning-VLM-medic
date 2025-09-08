import torch

def evaluate_utility(model, val_data):
    """评估通用能力"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_data:
            images, texts = batch["image"], batch["text"]
            true_label = int(batch["answer"])  # 期望为整数类别

            outputs = model(images, texts)  # 需要实现生成/分类逻辑以返回含logits的对象
            logits = outputs.logits
            pred_label = int(torch.argmax(logits, dim=-1).item())

            if pred_label == true_label:
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    print(f"Utility Test Accuracy: {accuracy:.4f}")
    return accuracy