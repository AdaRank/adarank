import itertools

from src.quality_meter.quality_measure import QualityMeasure


class Connectivity(QualityMeasure):

    def __init__(self, y, y_hat, max_val, physical_connections, simulated_projects):
        super().__init__(y, y_hat, max_val)
        self.simulated_projects = simulated_projects
        self.physical_connections = physical_connections


    def compute_quality(self):
        """
        * Computes how many component pairs in the simulated projects can be potentially connected (based on physical connections we already know)
        * This number is divided by the number of components in a project
        * The average over each project is returned
        :return:
        """
        scores = []
        phys_tuples = self.create_list_of_tuples_from_pc()
        for p in self.simulated_projects:
            connectable = 0
            for (c1, c2) in itertools.combinations(p, 2):
                if (c1, c2) in phys_tuples:
                    connectable += 1
            scores.append(connectable/len(p))
        return sum(scores)/(len(scores))

    def create_list_of_tuples_from_pc(self):
        t = []
        for p in self.physical_connections:
            temp = (p.get_anonymized_source_id(), p.get_anonymized_target_id())
            t.append(temp)
        return t
