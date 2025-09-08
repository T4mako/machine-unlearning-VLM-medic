from data.load_medmnist import prepare_datasets
from model.model_wrapper import QwenVLWithUnlearning
from train.trainer import EULTrainer
from eval import utility_eval, forgetting_eval, mia_eval


def main():
    # 1. 数据
    retain_data, forget_data, val_data = prepare_datasets()

    # 2. 模型
    model = QwenVLWithUnlearning()

    # 3. 训练
    trainer = EULTrainer(model, retain_data, forget_data, val_data)
    trainer.train(epochs=10)

    # 4. 评估
    utility_eval.evaluate_utility(model, val_data)
    forgetting_eval.evaluate_forgetting(model, forget_data)
    mia_eval.member_inference_attack(model, retain_data, forget_data)


if __name__ == "__main__":
    main()