from GlobalConstants import *


class BoardCell:
    def __init__(self, coordinates, neighbour_list):
        # Default boardcell has a filled value
        self.coordinates = coordinates
        self.value = (0,0)
        self.neighbour_list = neighbour_list

    def __repr__(self):
        return self.value

    def __str__(self):
        return self.value

    def set_value(self, new_value):
        self.value = new_value

    def get_value(self):
        return self.value

    def get_coordinates(self):
        return self.coordinates
    
    def get_neighbour_list(self):
        return self.neighbour_list
