from pytorch_lightning import seed_everything

from continual_learning.experiments.new_classes_mnist import run_model


def main():
    seed_everything(42)
    run_model()
    # run_experiment()

if __name__ == '__main__':
    main()
