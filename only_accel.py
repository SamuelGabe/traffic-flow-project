import numpy as np

def distance_to_next_car(cars:np.ndarray, n:int) -> int:
    """
    Returns the distance to the next car after n.

    Args:
        cars (np.array): The 1D array of cars at various points on the road
        n (int): The car for which we want to find the distance to the next car
    Returns:
        distance (int): The distance to the next car
    """
    distance = 1
    i = n + 1

    if i >= np.size(cars):
        i -= np.size(cars)

    # Repeat for every empty site
    while cars[i] == -1:
        distance += 1
        i += 1

        if i >= np.size(cars):
            i -= np.size(cars)
    
    return distance

if __name__ == '__main__':
    L = 10                                      # Amount of sites
    cars = np.ones(L, dtype=int) * -1           # Array with sites. If the site is -1, there is no car.
    cars[2] = 0                                 # Add cars to array
    cars[7] = 5                                 # Add cars to array
    vmax = 5                                    # Cars cannot go past this velocity
    number_of_iterations = 10
    
    # Keep track of positions of cars
    all_cars = np.ndarray((number_of_iterations+1, L), dtype=int)
    all_cars[0, :] = cars

    # Main loop
    for iteration in range(number_of_iterations):
        new_cars = np.ones(L, dtype=int) * -1   # Make new array so that we can update all the cars simultaneously

        # Update every car each iteration
        for n in np.arange(0, L):
            v = cars[n]

            # Don't update position if car isn't moving
            if v != -1:

                # Acceleration
                if v < vmax:
                    v += 1

                # Braking
                j = distance_to_next_car(cars, n)
                if v >= j:
                    v = j - 1
                
                # Randomisation
                randomisation = np.random.rand()
                if randomisation < p_slowdown:
                    v -= 1

                # Send cars back to start if they go past the end of the array
                if n + v < L:
                    new_cars[n+v] = v
                else:
                    new_cars[n+v-L] = v
            
            
        cars = new_cars
        all_cars[iteration+1] = cars
    
    # Remove spaces, change 0 to . and remove start and end brackets
    output = str(all_cars).replace(' ', '').replace('-1', '.')[1:-1]
    print(output)