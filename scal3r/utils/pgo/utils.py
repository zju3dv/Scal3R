import os
import pickle
import numpy as np


def matrix_to_similarity3(matrix: np.ndarray):
    """ Convert 4x4 Sim3 matrix to gtsam.Similarity3 (Georgia Tech Smoothing and Mapping) """
    from gtsam import Rot3, Point3, Similarity3
    scale = np.cbrt(np.linalg.det(matrix[:3, :3]))  # use cbrt for scale
    t = matrix[:3,  3] / scale
    R = matrix[:3, :3] / scale
    return Similarity3(Rot3(R), Point3(*t), scale)


class SerializableMixin:
    def save(self, filepath):
        """ Serialize the object's member variables to a file """
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, filepath):
        """ Deserialize member variables from a file into this object """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No such file: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data)
