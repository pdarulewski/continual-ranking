from continual_learning.experiments.new_classes_mnist import NewClassesMNIST


def main():
    experiment = NewClassesMNIST()
    experiment.run_model(
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    )


if __name__ == '__main__':
    main()
