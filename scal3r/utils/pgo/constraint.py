class Sim3Constraint:
    """ A class to represent a Sim3 constraint between two submaps """
    def __init__(
        self,
        submap_id1,
        submap_id2,
        sim3_matrix,
        overlap1,
        overlap2
    ):
        self.submap_id1 = submap_id1
        self.submap_id2 = submap_id2
        self.overlap1 = overlap1
        self.overlap2 = overlap2
        self.sim3 = sim3_matrix
