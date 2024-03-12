import numpy as np
import matplotlib.pyplot as plt

def distance_to_next_car(cars:np.ndarray, n:int, L) -> int:
    """
    Returns the distance to the next car after n.

    Args:
        cars (np.array): The 1D array of cars at various points on the road
        n (int): The car for which we want to find the distance to the next car
        L (int): Length of road
    Returns:
        distance (int): The distance to the next car
    """

    # Where the cars are (i.e. not -1)
    car_locations = np.where(cars != -1)[0]

    N_cars = np.size(car_locations)

    # If this car is the only car or there are no cars, return L
    if N_cars == 1 and car_locations[0] == n or N_cars == 0:
        return L
    
    # Distance to cars from car n
    distances = car_locations - n

    # we want the next car, i.e. the smallest positive value
    index = np.where(distances > 0, distances, 10*L).argmin()
    j = distances[index]

    # If it got to the end of the array and wrapped around. Avoids negative distance
    j += len(cars) if j < 0 else 0
    return j

def distance_to_previous_car(cars:np.ndarray, n:int, L) -> int:
    """
    Returns the distance to the next car after n.

    Args:
        cars (np.array): The 1D array of cars at various points on the road
        n (int): The car for which we want to find the distance to the next car
        L (int): Length of road
    Returns:
        distance (int): The distance to the next car
    """

        # Return large value if there are no cars
    if np.all(cars == -1):
        return 10*L

    # Where the cars are (i.e. not -1)
    car_locations = np.where(cars != -1)[0]
    
    # Distance to cars from car n. n - car_locations so that cars behind are positive distances
    distances = car_locations - n

    # we want the next car, i.e. the smallest positive value
    index = np.where(distances > 0, distances, 10*L).argmin()
    j = distances[index]

    # If it got to the end of the array and wrapped around. Avoids negative distance
    j += len(cars) if j < 0 else 0
    return j

def format_car_array(all_cars:np.ndarray) -> str:
    """Remove spaces, change 0 to . and remove start and end brackets"""
    return str(all_cars).replace(' ', '').replace('-1', '.')[1:-1]

# Calculate traffic flow for various densities with optional slow point in the middle
def calculate_cars(initial_cars:np.ndarray, iterations:int=10, vmax_fast:int=5, 
                           vmax_slow:int=2, p_slowdown:float=0.6, seed:int=0xDECAFBAD,
                           slowed=False):
    """
    Take a number of initial parameters and calculate how the cars will behave. 
    Args:
        initial_cars (np.ndarray): The initial array of how the cars are set up.
        num_of_iterations (int): Number of times to iterate by 1 time step
        vmax_fast (int): The maximum velocity a car can reach, default value 5
        vmax_slow (int): The maximum velocity a car can reach in a slow area, default value 2
        p_slowdown (float): The probability per timestep that a car will slow down
        seed (int): random seed for numpy
        slowed (bool): Whether or not there is a slow patch
    Returns:
        all_cars (np.ndarray): 2D array containing lots of cars
    """

    L = len(initial_cars)                       # Length of array
    np.random.seed(seed)                        # Seed to test with

    # Road of cars. Updates at the end of each iteration.
    cars = initial_cars

    # 3D array of all road positons for all iterations.
    all_cars = np.ndarray((iterations+1, L), dtype=int)
    all_cars[0, :] = cars

    # The start and end of the slow patch
    slow_patch = np.array([L*0.4, L*0.6])

    N_cars = np.count_nonzero(cars != -1)

    # Main loop
    for iteration in range(iterations):

        # Make new array so that we can update all the cars simultaneously
        new_cars = np.ones(L, dtype=int) * -1   

        # Set of random values used to slow down the cars.
        random_vals = np.random.rand(N_cars)

        # Which car is being iterated on. Every time a car is updated, this increases.
        current_car = 0

        # Update every car each iteration
        for n in np.arange(0, L):
            v = cars[n]
            
            # Decide vmax depending on if it's slow, if slowed=True
            vmax = vmax_slow if slowed and n+v >= slow_patch[0] and n+v <= slow_patch[1] else vmax_fast

            # Don't update position if car isn't moving
            if v != -1:
                # Acceleration
                if v < vmax:
                    v += 1

                # Braking
                j = distance_to_next_car(cars, n, L)
                if v >= j:
                    v = j - 1
                if v > vmax:
                    v = vmax
                
                # Randomisation
                if random_vals[current_car] < p_slowdown and v > 0:
                    v -= 1

                # Send cars back to start if they go past the end of the array
                if n + v < L:
                    new_cars[n+v] = v
                else:
                    new_cars[n+v-L] = v

                # Update to use the other random value
                current_car += 1
            
        # Update old array to be new array and add car positions to all_cars array
        cars = new_cars
        all_cars[iteration+1] = cars
    
    return all_cars

# Calculate traffic flow for various densities with slow point in the middle
def calculate_cars_lanes(initial_cars:np.ndarray, iterations:int=10, vmax_fast:int=5,
                           vmax_slow:int=2, p_slowdown:float=0.6, seed:int=0xDECAFBAD,
                           slowed=False):
    """
    Take a number of initial parameters and calculate how the cars will behave. Lanes have been implemented.
    Args:
        initial_cars (np.ndarray): The initial array of how the cars are set up.
        num_of_iterations (int): Number of times to iterate by 1 time step
        vmax_fast (int): The maximum velocity a car can reach, default value 5
        vmax_slow (int): The maximum velocity a car can reach in a slow area, default value 2
        p_slowdown (float): The probability per timestep that a car will slow down
        seed (int): random seed for numpy
        slowed (bool): Whether or not there is a slow patch
    Returns:
        all_cars (np.ndarray): 2D array containing lots of cars
    """
    L = np.shape(initial_cars)[0]
    np.random.seed(seed)
    
    cars = initial_cars

    all_cars = np.ones((iterations + 1, L, 2), dtype=int) * -1
    all_cars[0] = cars

    N_cars = np.count_nonzero(cars != -1)

    # Main loop
    for iteration in range(iterations):
        new_cars = np.ones((L, 2), dtype=int) * -1
        random_numbers = np.random.rand(N_cars)
        current_car = 0

        # Iterate for each car in lane
        for n in range(L):

            # l is the current lane.
            for l in range(2):
                not_l = 0 if l==1 else 1
                v = cars[n, l]
                if v != -1:
                    # ----Update speed----
                    # Accelerate
                    if v < vmax_fast:
                        v += 1

                    # Lane switch
                    j = np.zeros(2, dtype=int)
                    k = np.zeros(2, dtype=int)

                    # Takes the minimum distance out of the distance to the next car in the 
                    # old cars array and the distance to the next car in the new cars array.
                    j[l] = np.min([distance_to_next_car(cars[:, l], n, L), 
                                   distance_to_next_car(new_cars[:, l], n, L)])
                    
                    j[not_l] = np.min([distance_to_next_car(cars[:, not_l], n, L), 
                                         distance_to_next_car(new_cars[:, not_l], n, L)])
                    
                    # The line below is never used, so I removed it to improve efficiency
                    #k[l] = np.min([distance_to_previous_car(cars[:, l], n),
                    #               distance_to_previous_car(new_cars[:, l], n)])
                    
                    k[not_l] = np.min([distance_to_previous_car(cars[:, not_l], n, L),
                                       distance_to_previous_car(new_cars[:, not_l], n, L)])

                    """
                    Change lane if:
                        * Need to break

                    Then, don't change lane if:
                        * There is a car next to you
                        * A car in the other lane is about to overtake
                        * The distance to the car in this lane is smaller than the distance to 
                            the car in the next lane
                        * Don't need to change lane
                    """

                    switch_lane = False

                    # If the car needs to brake, attempt to switch lane.
                    if j[l] < v:
                        switch_lane = True

                    # If there is a car next to you, don't switch lane
                    if cars[n, not_l] != -1:
                        switch_lane = False
                    
                    # If a car in the other lane is going to overtake, don't switch
                    index_of_overtaking_car = n - k[not_l]

                    # While calculating the index, it could go negative if it's looped around
                    while index_of_overtaking_car < 0:
                        index_of_overtaking_car += L

                    velocity_of_overtaking_car = cars[index_of_overtaking_car, not_l]
                    if velocity_of_overtaking_car > k[not_l]:
                        switch_lane = False
                    
                    # If the current lane moves faster than the other lane, don't switch
                    if j[l] >= j[not_l]:
                        switch_lane = False


                    # Braking
                    j = j[not_l] if switch_lane else j[l]
                    if j <= v:
                        v = j - 1


                    # Randomization
                    if v > 0 and random_numbers[current_car] < p_slowdown:
                        v -= 1
                    
                    #----Update positions of cars----
                    current_car += 1
                
                    new_lane = not_l if switch_lane else l

                    # Send cars back to start if they go past the end of the array
                    if n + v < L:
                        new_cars[n+v, new_lane] = v
                    else:
                        new_cars[n+v-L, new_lane] = v
            
        # End of each iteration
        cars = new_cars
        all_cars[iteration + 1] = cars
    
    # End of function
    return all_cars

# Calculate traffic flow for various densities
def calculate_flows(density_range:tuple=(0, 1), points=150, 
                    L:int=400, iterations:int=1000, slowed:bool=False, 
                    lanes:bool=False, seed=0xDECAFBAD, vmax=5) -> np.ndarray:
    """
    Creates a flow diagram of density vs flow when provided with initial cars.
    Args:
        density_range (tuple): Range of densities of cars on road
        points (int): How many densities to plot
        L (int): Length of road
        iterations (int): How many iterations for each density
        slowed (bool): Whether or not to have a slow patch
        lanes (bool): 2 lanes if True, 1 lane if False
        seed (int): The seed to use to decide random slowing down
        vmax (int): Maximum velocity of cars
    Returns:
        x (np.ndarray): Array containing values of density
        flow (np.ndarary): Array containing the flow for each value of x
    """

    np.random.seed(seed)
    
    x = np.linspace(density_range[0], density_range[1], points)
    
    # Set of all sets of iterations over time.
    if lanes:
        # Dimension 1: density. 2: iteration. 3: cell. 4: lane.
        all_cars_set = np.ndarray((points, iterations+1, L, 2), dtype=int)
    else:
        all_cars_set = np.ndarray((points, iterations+1, L), dtype=int)

    # For each density, create all_cars array and add to all_cars_set
    for n, density in enumerate(x):
        # ----Initial Cars Array----

        # Number of cars required
        N_cars = np.floor(L * density).astype(int)

        # Empty cars array
        lane = np.array([0 if n < N_cars else -1 for n in range(L)])

        # ----Calculate All Cars----
        # If we want lanes, use the calculate_cars_lanes method. Construct 2D array.
        if lanes:
            initial_cars = np.repeat(np.expand_dims(lane, axis=-1), 2, axis=-1)

            all_cars_set[n] = calculate_cars_lanes(
                initial_cars, 
                iterations=iterations, 
                slowed=slowed, 
                vmax_fast=vmax, 
                seed=seed
            )
        
        
        # Otherwise, just use one lane and use the calculate_cars method.
        else:
            all_cars_set[n] = calculate_cars(
                lane, 
                iterations=iterations, 
                slowed=slowed, 
                vmax_fast=vmax, 
                seed=seed
            )
    
    # ----Create Output Array----

    # Minimum number of iterations from which to calculate traffic flow.
    flow_min_iterations = np.floor(iterations * 0.6).astype(int)

    velocities_set = np.clip(all_cars_set, a_min=0, a_max=None)[:, flow_min_iterations:]

    # Divide by 2 if there are two lanes.
    if not lanes:
        flows = np.sum(velocities_set, axis=-1) / L
    else:
        flows = np.sum(velocities_set, axis=(-1, -2)) / L / 2

    mean_flows = np.mean(flows, axis=-1)
    std = np.std(flows, axis=-1)

    # Transpose so the shape is [POINTS, N_SEEDS] instead of [N_SEEDS, POINTS].
    mean_flows = np.transpose(mean_flows)
    std = np.transpose(std)

    return x, mean_flows, std
