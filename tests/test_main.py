import numpy as np
import pytest
from traffic import *

def test_acceleration():
    initial_cars = np.array([-1, 0, -1, -1, -1, -1, -1, -1, -1, -1])
    all_cars = calculate_cars(initial_cars, iterations=6, vmax_fast=5, p_slowdown=0.0)
    
    assert np.all(all_cars == np.array([[-1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, 2, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, -1, 3, -1, -1],
                                 [-1, 4, -1, -1, -1, -1, -1, -1, -1, -1],
                                 [-1, -1, -1, -1, -1, -1, 5, -1, -1, -1],
                                 [-1, 5, -1, -1, -1, -1, -1, -1, -1, -1]]))

def test_braking():
    initial_cars = np.array([-1, -1, 5, -1, -1, -1, 0, -1, -1, -1])
    all_cars = calculate_cars(initial_cars, iterations=1, vmax_fast=5, p_slowdown=0.0)
    assert np.all(all_cars == np.array([[-1, -1, 5, -1, -1, -1, 0, -1, -1, -1],
                                        [-1, -1, -1, -1, -1, 3, -1, 1, -1, -1]]))
    

    initial_cars = np.array([-1, -1, 0, -1, -1, -1, -1, 5, -1, -1])
    all_cars = calculate_cars(initial_cars, iterations=1, vmax_fast=5, p_slowdown=0.0)
    assert np.all(all_cars == np.array([[-1, -1, 0, -1, -1, -1, -1, 5, -1, -1],
                                        [-1, 4, -1, 1, -1, -1, -1, -1, -1, -1]]))

def test_random():
    initial_cars = np.array([-1, -1, 5, -1, -1, -1, 0, -1, -1, -1])
    all_cars = calculate_cars(initial_cars, iterations=10, p_slowdown=0.6, seed=0xDECAFBAD)
    assert np.all(all_cars == np.array([[-1, -1, +5, -1, -1, -1, +0, -1, -1, -1],
                                        [-1, -1, -1, -1, -1, +3, +0, -1, -1, -1],
                                        [-1, -1, -1, -1, -1, +0, -1, 1, -1, -1],
                                        [-1, -1, -1, -1, -1, -1, +1, -1, +1, -1],
                                        [+2, -1, -1, -1, -1, -1, +0, -1, -1, -1],
                                        [-1, -1, +2, -1, -1, -1, +0, -1, -1, -1],
                                        [-1, -1, -1, -1, -1, +3, +0, -1, -1, -1],
                                        [-1, -1, -1, -1, -1, +0, -1, +1, -1, -1],
                                        [-1, -1, -1, -1, -1, +0, -1, -1, +1, -1],
                                        [+2, -1, -1, -1, -1, -1, +1, -1, -1, -1],
                                        [-1, -1, -1, +3, -1, -1, -1, +1, -1, -1]]))

def test_checkpoint():
    initial_cars = np.ones(100, dtype=int) * -1
    initial_cars[:10] = 0
    all_cars = calculate_cars(initial_cars, 
                              iterations=150, 
                              p_slowdown=0.6, 
                              vmax_fast=5, 
                              seed=0xDECAFBAD
    )
    
    DESIRED_RESULT = '[......................4.......5........5.................4.......5.........10..1....2.....2.........]'
    assert str(all_cars[-1]).replace('-1', '.').replace(' ', '').replace('\n', '') == DESIRED_RESULT

def test_lanes():
    initial_cars = np.array([[+5, -1],
                             [-1, -1],
                             [+2, -1],
                             [-1, +3],
                             [-1, -1],
                             [-1, -1],
                             [-1, -1],
                             [-1, -1],
                             [-1, -1],
                             [-1, -1],
                             [-1, -1],
                             [-1, -1]])

    all_cars = calculate_cars_lanes(
        initial_cars,
        iterations=4,
        p_slowdown=0.,
        vmax_fast=5
    )

    """
    [.......5....]
    [..5.....4...]
    """

    desired_result = np.ones((12, 2), dtype=int) * -1
    desired_result[7, 0]  = 5
    desired_result[2, 1]  = 5
    desired_result[8, 1] = 4

    assert np.all(all_cars[-1] == desired_result)