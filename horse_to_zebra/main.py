from core.settings import Settings
from load_data import LoadData
from losses import LossFunction
from train import TrainModel


def main():
    settings = Settings()
    load_data_instance = LoadData(settings)
    loss_function_instance = LossFunction(settings)
    train_model = TrainModel(settings, load_data_instance, loss_function_instance)
    train_model.train_loop()


if __name__ == "__main__":
    main()
