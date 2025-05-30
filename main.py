import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from skimage.measure import label

# ===================== Helper Functions =====================
def indexer(i, j, mesh_x):
    """Convert 2D index to 1D index (0-based)"""
    return (i-1)*mesh_x + j - 1  # -1 for Python 0-based indexing

def compute_directional_averages(matrix):
    """Calculate thermal conductivity averages in four directions"""
    rows, cols = matrix.shape
    averages = np.zeros((rows, cols, 4))
    
    for i in range(rows):
        for j in range(cols):
            # Up direction
            if i == 0:
                averages[i, j, 0] = matrix[i, j] / 2
            else:
                averages[i, j, 0] = (matrix[i, j] + matrix[i-1, j]) / 2
            
            # Down direction
            if i == rows-1:
                averages[i, j, 1] = matrix[i, j] / 2
            else:
                averages[i, j, 1] = (matrix[i, j] + matrix[i+1, j]) / 2
            
            # Left direction
            if j == 0:
                averages[i, j, 2] = matrix[i, j] / 2
            else:
                averages[i, j, 2] = (matrix[i, j] + matrix[i, j-1]) / 2
            
            # Right direction
            if j == cols-1:
                averages[i, j, 3] = matrix[i, j] / 2
            else:
                averages[i, j, 3] = (matrix[i, j] + matrix[i, j+1]) / 2
                
    return averages

def calculate_temperature_matrix(L, a, q, T0, mesh_x, mesh_y, given_distribution, epsilon):
    """Calculate temperature field matrix"""
    A = lil_matrix((mesh_x*mesh_y, mesh_x*mesh_y))
    b = np.zeros(mesh_x*mesh_y)
    T = np.ones(mesh_x*mesh_y) * T0

    delta_x = L / mesh_x
    delta_y = L / mesh_y
    delta_x_square = delta_x**2
    delta_y_square = delta_y**2

    start_index = int((L - a) / 2 / delta_x)
    end_index = int((L + a) / 2 / delta_x)

    k_averages = compute_directional_averages(given_distribution)

    for i in range(mesh_x):
        for j in range(mesh_y):
            idx = indexer(i, j, mesh_x)

            if i == 0:
                A[idx, idx] = 1
                A[idx, indexer(i+1, j, mesh_x)] = -1
                b[idx] = 0
            elif i == mesh_x-1:
                A[idx, idx] = 1
                A[idx, indexer(i-1, j, mesh_x)] = -1
                b[idx] = 0
            elif j == 0 and i > 0 and i < mesh_x-1:
                A[idx, idx] = 1
                A[idx, indexer(i, j+1, mesh_x)] = -1
                b[idx] = 0
            elif j == mesh_y-1 and i > 0 and i < mesh_x-1:
                A[idx, idx] = 1
                A[idx, indexer(i, j-1, mesh_x)] = -1
                b[idx] = 0
            else:
                S = (k_averages[i, j, 1] + k_averages[i, j, 0]) / delta_x_square + \
                    (k_averages[i, j, 2] + k_averages[i, j, 3]) / delta_y_square
                A[idx, idx] = 1
                A[idx, indexer(i-1, j, mesh_x)] = -k_averages[i, j, 0] / S / delta_x_square
                A[idx, indexer(i+1, j, mesh_x)] = -k_averages[i, j, 1] / S / delta_x_square
                A[idx, indexer(i, j-1, mesh_x)] = -k_averages[i, j, 2] / S / delta_y_square
                A[idx, indexer(i, j+1, mesh_x)] = -k_averages[i, j, 3] / S / delta_y_square
                b[idx] = q / S

    for i in range(start_index, end_index+1):
        idx = indexer(mesh_x-1, i, mesh_x)
        A[idx, :] = 0
        A[idx, idx] = 1
        b[idx] = T0

    A = A.tocsr()
    T = spsolve(A, b)

    return A, b, T

def adjust_high_k(individual, num_high_k):
    """Adjust number of high conductivity cells"""
    total_high_k = np.sum(individual)
    if total_high_k > num_high_k:
        idx1 = np.where(individual == 1)[0]
        remove_count = int(total_high_k - num_high_k)  # Ensure integer
        if len(idx1) > 0 and remove_count > 0:  # Add safety check
            remove_indices = np.random.choice(idx1, remove_count, replace=False)
            individual[remove_indices] = 0
    elif total_high_k < num_high_k:
        idx0 = np.where(individual == 0)[0]
        add_count = int(num_high_k - total_high_k)  # Ensure integer
        if len(idx0) > 0 and add_count > 0:  # Add safety check
            add_indices = np.random.choice(idx0, add_count, replace=False)
            individual[add_indices] = 1
    return individual

def calculate_fitness_with_penalty(T_flipped, k_opt):
    """Calculate fitness function with penalty term"""
    # Calculate average temperature
    mean_temp = np.mean(T_flipped)
    
    # Calculate connectivity penalty term
    k_binary = k_opt > 250
    cc = label(k_binary, connectivity=2)
    num_components = np.max(cc)
    
    # Dynamic penalty coefficient (based on number of connected components)
    if num_components <= 2:
        penalty_coeff = 0.0005  # Light penalty
    else:
        penalty_coeff = 0.002  # Stronger penalty
    
    # Total fitness = average temperature + connectivity penalty
    return mean_temp + penalty_coeff * (num_components - 1)

# ===================== Main Program =====================
if __name__ == "__main__":
    # Initialize
    plt.close('all')

    # Parameter settings (cannot be changed!)
    Lx = 1.0
    Ly = Lx
    nx = 51
    ny = 51
    k0 = 1.0
    k1 = 500.0
    T0 = 300.0
    q = 100.0
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Read initial thermal conductivity distribution
    try:
        k = pd.read_csv('Your csv file', header=None).to_numpy()
    except FileNotFoundError:
        print("Warning: k.csv not found, using zeros as initial distribution")
        k = np.zeros((nx, ny))
    
    print(f"Initial number of high conductivity cells: {np.sum(k > 250)}")

    # Basic parameters
    total_cells = nx * ny  # Total number of cells
    num_high_k = int(total_cells * 0.15)  # Ensure integer for number of high conductivity cells (15%)

    # Initial distribution (multiple vertical paths)
    k_heuristic = np.zeros((nx, ny))
    num_paths = 5  # 5 vertical paths
    path_width = max(1, int(num_high_k / (nx * num_paths)))  # Ensure at least 1 and integer
    path_spacing = max(1, int(nx / (num_paths + 1)))  # Ensure at least 1 and integer

    for p in range(num_paths):
        start_col = (p) * path_spacing + path_spacing - path_width
        if start_col + path_width <= ny:  # Ensure we don't exceed array bounds
            k_heuristic[:, start_col:start_col+path_width] = 1  # Fill vertical paths

    k_heuristic = k_heuristic.flatten()
    k_heuristic = adjust_high_k(k_heuristic, num_high_k)  # Adjust to exact number
    current_solution = k_heuristic.copy()

    # Calculate initial temperature
    k_opt = current_solution.reshape((nx, ny)) * (k1 - k0) + k0
    tolerance = 1.0e-5
    A, b, T = calculate_temperature_matrix(Lx, 0.1*Lx, q, T0, nx, ny, k_opt, tolerance)
    T_reshaped = T.reshape((nx, ny))
    T_flipped = np.flipud(T_reshaped.T)
    T_flipped = k0 * (T_flipped - T0) / Lx**2 / q
    current_fitness = calculate_fitness_with_penalty(T_flipped, k_opt)
    print(f'Initial average dimensionless temperature: {current_fitness:.6f}')

    # Improved greedy algorithm (with enhanced continuity constraints)
    max_iter_greedy = 10000  # Increased iterations
    num_swaps = 20  # Increased number of swaps per attempt
    best_solution_greedy = current_solution.copy()
    best_fitness_greedy = current_fitness
    fitness_history_greedy = np.zeros(max_iter_greedy)

    # Temperature field sensitivity analysis (to guide swaps)
    temp_sensitivity = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            temp_sensitivity[i,j] = T_flipped[i,j]

    temp_sensitivity = temp_sensitivity / np.max(temp_sensitivity)  # Normalize

    for iter in range(max_iter_greedy):
        # Try multiple swaps, select the best
        idx1 = np.where(current_solution == 1)[0]  # High conductivity cell indices
        idx0 = np.where(current_solution == 0)[0]  # Low conductivity cell indices
        best_swap_fitness = current_fitness
        best_swap_solution = current_solution.copy()
        
        for s in range(num_swaps):
            # Select swap positions based on temperature sensitivity
            if np.random.rand() < 0.7:  # 70% probability to select high temperature regions
                sorted_idx = np.argsort(T_flipped.flatten())[::-1]
                candidate_pos = sorted_idx[:int(0.2*nx*ny)]  # Top 20% high temperature regions
                candidate_pos = np.intersect1d(candidate_pos, idx0)
                if len(candidate_pos) > 0:
                    swap0 = np.random.choice(candidate_pos)
                    swap1 = np.random.choice(idx1)
                else:
                    swap1 = np.random.choice(idx1)
                    swap0 = np.random.choice(idx0)
            else:
                swap1 = np.random.choice(idx1)
                swap0 = np.random.choice(idx0)
            
            new_solution = current_solution.copy()
            new_solution[swap1] = 0
            new_solution[swap0] = 1

            # Calculate new thermal conductivity distribution
            k_opt = new_solution.reshape((nx, ny)) * (k1 - k0) + k0
            
            # Calculate temperature field and fitness (with penalty)
            A, b, T = calculate_temperature_matrix(Lx, 0.1*Lx, q, T0, nx, ny, k_opt, tolerance)
            T_reshaped = T.reshape((nx, ny))
            T_flipped_new = np.flipud(T_reshaped.T)
            T_flipped_new = k0 * (T_flipped_new - T0) / Lx**2 / q
            new_fitness = calculate_fitness_with_penalty(T_flipped_new, k_opt)

            # Select better solution
            if new_fitness < best_swap_fitness:
                best_swap_fitness = new_fitness
                best_swap_solution = new_solution.copy()
                T_flipped = T_flipped_new.copy()

        # Accept best swap
        current_solution = best_swap_solution.copy()
        current_fitness = best_swap_fitness

        # Update global best solution
        if current_fitness < best_fitness_greedy:
            best_solution_greedy = current_solution.copy()
            best_fitness_greedy = current_fitness

        fitness_history_greedy[iter] = best_fitness_greedy
        
        # Calculate current number of connected components
        k_binary = best_solution_greedy.reshape((nx, ny)) > 250
        cc = label(k_binary, connectivity=2)
        num_components = np.max(cc)
        print(f'Greedy iteration {iter}, best fitness: {best_fitness_greedy:.6f}, connected components: {num_components}')

        # Early termination condition
        if best_fitness_greedy < 0.0035 and num_components <= 2:
            print('Target reached: fitness < 0.0035 and high conductivity regions connected')
            break
        
        # Dynamic adjustment of swap strategy
        if iter % 100 == 0:
            num_swaps = min(30, num_swaps + 2)  # Gradually increase number of swaps

    # Final optimal solution
    k_opt = best_solution_greedy.reshape((nx, ny)) * (k1 - k0) + k0
    print(f"Optimized number of high conductivity cells: {np.sum(k_opt > 250)}")

    # Improved smoothing: ensure continuous distribution of high conductivity regions
    k_binary = k_opt > 250  # Convert to binary matrix, high conductivity regions as 1

    # Find largest connected region
    cc = label(k_binary, connectivity=2)
    num_components = np.max(cc)
    if num_components > 0:
        largest_component = np.argmax(np.bincount(cc.flat)[1:]) + 1
        k_smooth = cc == largest_component
    else:
        k_smooth = np.zeros_like(k_binary)

    # Add isolated points adjacent to main region (considering temperature sensitivity)
    temp_sensitivity_binary = temp_sensitivity > 0.7  # High temperature regions
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            if not k_smooth[i,j] and k_binary[i,j] and temp_sensitivity_binary[i,j]:
                # Check if adjacent to main region
                neighbors = k_smooth[i-1:i+2, j-1:j+2]
                if np.sum(neighbors) > 0:
                    k_smooth[i,j] = True

    # Adjust number of high conductivity cells to remain constant
    k_smooth = adjust_high_k(k_smooth.flatten(), num_high_k)
    k_smooth = k_smooth.reshape((nx, ny))
    k_opt = k_smooth * (k1 - k0) + k0  # Convert back to conductivity values
    print(f"Number of high conductivity cells after smoothing: {np.sum(k_opt > 250)}")

    # Save optimized thermal conductivity distribution to Excel file
    output_filename = 'optimized_k_distribution.csv'
    pd.DataFrame(k_opt).to_csv(output_filename, index=False, header=False)
    print(f'Optimized thermal conductivity distribution saved to: {output_filename}')


    # Plot smoothed thermal conductivity distribution
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, np.flipud(k_opt), levels=50, cmap='jet')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.title('Smoothed Thermal Conductivity Distribution', fontsize=16)
    plt.xlabel('$x$', fontsize=18)
    plt.ylabel('$y$', fontsize=18)
    plt.show()

    # Calculate and plot smoothed temperature distribution
    tolerance = 1.0e-4
    A, b, T = calculate_temperature_matrix(Lx, 0.1*Lx, q, T0, nx, ny, k_opt, tolerance)
    T_reshaped = T.reshape((nx, ny))
    T_flipped = np.flipud(T_reshaped.T)
    T_flipped = k0 * (T_flipped - T0) / Lx**2 / q
    final_fitness = calculate_fitness_with_penalty(T_flipped, k_opt)
    print(f"Average dimensionless temperature after smoothing: {np.mean(T_flipped):.6f}")
    print(f"Final fitness value: {final_fitness:.6f}")

    # plt.figure(figsize=(8, 6))
    # plt.contourf(X, Y, T_flipped, levels=50, cmap='jet')
    # plt.colorbar()
    # plt.gca().set_aspect('equal')
    # plt.title('Smoothed Dimensionless Temperature Distribution', fontsize=16)
    # plt.xlabel('$x$ (m)', fontsize=18)
    # plt.ylabel('$y$ (m)', fontsize=18)
    # plt.show()

    # Plot fitness history
    plt.figure()
    plt.plot(range(iter+1), fitness_history_greedy[:iter+1], 'b-', linewidth=2, label='Greedy')
    plt.xlabel('Iterations')
    plt.ylabel('Best Fitness')
    plt.title('Optimization Process')
    plt.legend()
    plt.grid(True)
    plt.show()