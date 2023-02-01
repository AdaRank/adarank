import os


class Writer:
    """
    Writes simulated projects to txt-file
    """
    def __init__(self, path, title, simulator):
        self.title = title
        self.simulator = simulator
        self.path = path

    def write_output(self):
        # write simulated projects to file
        with open(os.path.join(self.path, self.title), 'w', encoding='UTF8') as f:
            for i, seq in enumerate(self.simulator.sequences):
                temp = ""
                for el in seq:
                    temp += str(el) + " -1 "
                temp += "-2"
                f.write(temp)
                f.write("\n")