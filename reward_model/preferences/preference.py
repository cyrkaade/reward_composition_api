from reward_composition_api.data_structures import Trajectory

class Preference:
    def __init__(self, t1: Trajectory, t2: Trajectory, rating: float):
        self.t1 = t1
        self.t2 = t2
        self.rating = rating
