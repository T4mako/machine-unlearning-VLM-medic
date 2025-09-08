import torch

def evaluate_forgetting(model, forget_data):
    """评估遗忘效果"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in forget_data:
            images, texts = batch["image"], batch["text"]
            true_label = int(batch["answer"])  # 期望为整数类别

            outputs = model(images, texts)
            logits = outputs.logits
            pred_label = int(torch.argmax(logits, dim=-1).item())

            # 注意：我们希望模型在这里答错！
            if pred_label == true_label:
                correct += 1
            total += 1

    accuracy = correct / max(total, 1)
    print(f"Forgetting Accuracy: {accuracy:.4f} (Lower is better!)")
    return accuracy