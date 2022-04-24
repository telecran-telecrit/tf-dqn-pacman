from .start import PATH_PLOTS, PATH_DATA

from matplotlib import pyplot as plt
import statistics
import pickle


class Atom:
    """
    Simple class to hold data and compute mean and total of values.
    """

    def __init__(self):
        self._raw = []
        self._mean = []
        self._total = []

    def mean(self):
        self._mean.append(statistics.mean(self._raw))

    def total(self):
        self._total.append(sum(self._raw))

    def clear(self):
        self._raw.clear()

    def append(self, item):
        self._raw.append(item)


class Structure:
    """
    Class to hold all (dynamic) data
    """

    def __init__(self):
        self.ep = 1
        self.successes = 0
        self.losses = Atom()
        self.rewards = Atom()
        self.q_values = Atom()

    def __iter__(self):
        yield self.losses._raw
        yield self.rewards._mean
        yield self.q_values._mean
        yield self.rewards._raw
        yield self.rewards._total
        yield self.rewards._total

    def round(self):
        self.ep += 1
        self.rewards.mean()
        self.rewards.total()
        self.q_values.mean()
        self.q_values.total()
        self.clear()

    def clear(self):
        self.losses.clear()
        self.rewards.clear()
        self.q_values.clear()

    def save(self):
        """Save data in `pickle` file"""
        with open(PATH_DATA / f"episode-{self.ep}.pkl", "wb") as file:
            pickle.dump(list(self) + [self.successes], file)
        print(f"Episode {self.ep} saved.")


class Display:

    """
    Class to display data
    """

    Y_LABELS = (
        "Loss per optimization",
        "Average of rewards per episode",
        "Average of max predicted Q value",
        "Rewards per action",
        "Total of rewards per episode",
        "Total of max predicted Q value",
    )

    def __init__(self, dynamic=True, image=False):
        self.data = Structure()
        self.dynamic = dynamic
        if dynamic:  # To display the game and progression of the network's performance
            self.fig = plt.figure(figsize=(20, 8))
            m = [2, 2, 2, 2, 2, 2, 1]
            n = [1, 2, 3, 5, 6, 7, 4]
            self.axis = [self.fig.add_subplot(i, 4, j) for i, j in zip(m, n)]
            self.fig.tight_layout()
        else:  # To display only network's performance
            self.fig, self.axis = plt.subplots(2, 3, figsize=(16, 10))
            self.fig.tight_layout()
            self.axis = self.axis.flatten()
        self.save = self.save_all if image else self.save_data

    def update_axis(self, observation=False):
        for axis, data in (self.axis[:-1], self.data):
            axis.plot(range(len(data)), data)
        for label, axis in zip(self.Y_LABELS, self.axis[:-1]):
            axis.set_ylabel(label)
        if observation:
            self.axis[6].imshow(self.obs)
        self.fig.suptitle(f"Episode {self.data.ep} | Total of successes = {self.successes}")

    def show(self):
        plt.ion()
        self.update_axis(True)
        plt.draw()
        plt.pause(0.0001)
        for axis in self.axis:
            axis.cla()

    def save(self):
        """Save data in `pickle` file and an image"""
        self.update_axis()
        self.fig.tight_layout()
        plt.savefig(PATH_PLOTS / f"episode-{self.data.ep}.png")
        print(f"Figure {self.data.ep} saved.")
        for axis in self.axis:
            axis.cla()
        self.data.save()