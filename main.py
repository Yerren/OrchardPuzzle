import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections.abc import Iterable
from collections.abc import Collection
import pandas as pd
import time


def check_overlap(config_1: np.ndarray, config_2: np.ndarray) -> bool:
    """
    Checks whether two configurations overlap.
    
    :param config_1: An (8, 8) numpy array representing a configuration.
    :param config_2: An (8, 8) numpy array representing a configuration.
    :return: Whether config_1 and config_2 overlap.
    :rtype: bool
    """
    return np.any((config_1 & config_2) != 0)


def check_if_config_valid(config_in: np.ndarray, config_collection: Collection[np.ndarray]) -> bool:
    """
    Checks whether config_in is a valid configuration. To be valid, it must not overlap with the house, and also not
    already exist in config_collection.

    :param config_in: A configuration to validate.
    :param config_collection: A collection of existing configurations which config_in must not match.
    :return: Whether config_in is a valid configuration.
    """
    if check_overlap(config_in, house_config):
        return False
    for existing_config in config_collection:
        if (config_in == existing_config).all():
            return False

    return True


def apply_all_transformations(configs_in: Iterable[np.ndarray]) -> np.ndarray:
    """
    Applies all valid transformations (translation, flips, and rotations) to all configurations provided. Ensures that
    no configuration is repeated.

    This function assumes that 7x7 transformations are created such that they strictly span only the first 7 rows and
    columns (i.e., have no entries in the 8th row or column).

    :param configs_in: A 3-dimensional numpy array (or list of 2-dimensional numpy arrays), of shape (num_configs, 8, 8)
        that contains a set of configurations to apply the transformations to.
    :return: A 3-dimensional numpy array, in the same format as configs_in, containing all the (transformed and
        original) configurations.
    """

    # Translate 7x7s
    translated_configs = []
    for config in configs_in:
        if check_if_config_valid(config, translated_configs):
            translated_configs.append(config)
        rows, cols = np.where(config)
        d_translate = GRID_SIZE - 1 - (np.max(rows))
        r_translate = GRID_SIZE - 1 - (np.max(cols))

        for r_shift in range(r_translate + 1):
            for d_shift in range(d_translate + 1):
                if r_shift == d_shift == 0:
                    continue

                shifted_config = np.roll(config, d_shift, axis=0)
                shifted_config = np.roll(shifted_config, r_shift, axis=1)

                if check_if_config_valid(shifted_config, translated_configs):
                    translated_configs.append(shifted_config)

    # Flips (don't need to do lr+ud, as that is the same as a 180-degree rotation.
    flipped_configs = translated_configs.copy()
    for config in translated_configs:
        lr_flip = np.fliplr(config)
        if check_if_config_valid(lr_flip, flipped_configs):
            flipped_configs.append(lr_flip)

        ud_flip = np.flipud(config)
        if check_if_config_valid(ud_flip, flipped_configs):
            flipped_configs.append(ud_flip)

    # Rotations
    rotated_configs = flipped_configs.copy()
    for config in flipped_configs:
        rot90 = config
        for _ in range(3):
            rot90 = np.rot90(rot90)
            if check_if_config_valid(rot90, rotated_configs):
                rotated_configs.append(rot90)

    return np.array(rotated_configs)


def plot_solution(configs_in: Iterable[np.ndarray]) -> None:
    """
    Plots up to four provided configurations in different colours.
    
    :param configs_in: The configuration(s) to plot.
    """
    fig, ax = plt.subplots()

    # Create a custom colormap for each color
    cmap_1 = np.array([[0, 0, 0, 0], [165 / 256, 25 / 256, 25 / 256, 1]])
    cmap_2 = np.array([[0, 0, 0, 0], [200 / 256, 170 / 256, 20 / 256, 1]])
    cmap_3 = np.array([[0, 0, 0, 0], [30 / 256, 130 / 256, 50 / 256, 1]])
    cmap_4 = np.array([[0, 0, 0, 0], [65 / 256, 100 / 256, 165 / 256, 1]])
    cmap_house = np.array([[0, 0, 0, 0], [150 / 256, 150 / 256, 150 / 256, 1]])

    # Create new colormaps
    cmap_list = [ListedColormap(cmap_1), ListedColormap(cmap_2), ListedColormap(cmap_3), ListedColormap(cmap_4)]

    cmap_house = ListedColormap(cmap_house)

    # Overlay the arrays on top of each other in different colors, with the house always included at the end.
    for config, cmap in zip(configs_in, cmap_list):
        ax.imshow(config, cmap=cmap)
    ax.imshow(house_config, cmap=cmap_house)

    plt.show()


def check_and_add_to_memoization(config_in: np.ndarray) -> bool:
    """
    Checks whether config_in has already been seen (based on whether it is in visited_nodes). If so, it returns True.
    Otherwise, it adds config_in (and it's left-right mirror) to visited_nodes, and returns True.
    :param config_in: The configuration to check whether it has already been seen.
    :return: Whether the configuration has already been seen.
    """
    global visited_nodes
    global num_operations

    num_operations += 1

    if np.array2string(config_in) in visited_nodes:
        return True
    else:
        visited_nodes.add(np.array2string(config_in))
        visited_nodes.add(np.array2string(np.fliplr(config_in)))
        return False


def check_combs_iterative_brute_force(configs_in: Collection[np.ndarray], memoize_depth: int = 0) -> tuple | None:
    """
    Iteratively checks all possible configurations using brute force. Very inefficient, but simple.

    :param configs_in: All valid configurations to check (including transformations).
    :param memoize_depth: Up to what depth of the search tree to apply memoization to.
    :return: A tuple containing the solution configurations, or None, if no solution is found.
    """
    global num_operations
    global visited_nodes

    for choice_1 in configs_in:
        if memoize_depth > 0 and check_and_add_to_memoization(choice_1):
            continue
        num_operations += 1
        if (house_config + choice_1).sum() != 14:
            continue

        for choice_2 in configs_in:
            if memoize_depth > 1 and check_and_add_to_memoization(choice_1+choice_2):
                continue
            num_operations += 1
            if (house_config + choice_1 + choice_2).sum() != 24:
                continue

            for choice_3 in configs_in:
                if memoize_depth > 2 and check_and_add_to_memoization(choice_1 + choice_2 + choice_3):
                    continue
                num_operations += 1
                if (house_config + choice_1 + choice_2 + choice_3).sum() != 34:
                    continue

                for choice_4 in configs_in:
                    if memoize_depth > 3 and check_and_add_to_memoization(choice_1 + choice_2 + choice_3 + choice_4):
                        continue
                    num_operations += 1
                    output = house_config + choice_1 + choice_2 + choice_3 + choice_4
                    if output.sum() == 44:
                        return tuple([choice_1, choice_2, choice_3, choice_4])

    return None


def check_combs_recursive_brute_force(configs_in: Collection[np.ndarray], current_config: np.ndarray,
                                      num_configs: int = 0, selected_configs: tuple = (),
                                      memoize_depth: int = 0) -> tuple | None:
    """
    Recursively checks all possible configurations using brute force. Inefficient, but simple.

    :param configs_in: All valid configurations to check (including transformations).
    :param current_config: The current configuration, consisting of all the selected configs added together.
    :param num_configs: The number of configurations already selected.
    :param selected_configs: A tuple of the configurations selected so far.
    :param memoize_depth: Up to what depth (as represented by num_configs) of the search tree to apply memoization to.
    :return: A tuple containing the solution configurations, or None, if no solution is found.
    """
    global num_operations

    # If we have seen the current configuration before, we don't need to check again
    if memoize_depth != 0 and num_configs < memoize_depth + 1 and check_and_add_to_memoization(current_config):
        return None

    num_operations += 1
    # Don't go deeper in the search if this configuration already is invalid
    if current_config.sum() != 4 + 10 * num_configs:
        return None
    elif num_configs == 4:
        # If we are at the final layer of the search tree, return whether the solution is valid.
        return selected_configs

    for config in configs_in:
        result = check_combs_recursive_brute_force(configs_in, current_config + config, num_configs + 1,
                                                   selected_configs + tuple([config]), memoize_depth=memoize_depth)
        if result is not None:
            return result

    return None


if __name__ == "__main__":
    GRID_SIZE = 8

    """
    Set up the initial valid configurations manually, using the logic described in the readme.
    """
    configs = np.zeros((5, 8, 8), dtype='bool')
    config_indices = np.array([[[7, 0], [5, 1], [0, 2], [3, 2], [5, 2], [6, 2], [1, 3], [5, 4], [4, 6], [5, 7]],
                               [[0, 0], [0, 3], [0, 5], [0, 6], [1, 4], [2, 2], [3, 0], [4, 1], [5, 0], [6, 0]],
                               [[0, 1], [1, 3], [2, 5], [3, 1], [3, 2], [3, 4], [3, 7], [5, 1], [6, 1], [7, 0]],
                               [[0, 3], [2, 2], [2, 4], [3, 3], [4, 1], [4, 2], [4, 4], [4, 5], [6, 0], [6, 6]],
                               [[0, 0], [3, 0], [3, 3], [4, 2], [5, 0], [5, 1], [5, 4], [5, 5], [6, 0], [6, 6]]])

    house_config = np.zeros((8, 8), dtype='bool')
    house_config[6, 3] = True
    house_config[6, 4] = True
    house_config[7, 3] = True
    house_config[7, 4] = True

    for i, indices in enumerate(config_indices):
        configs[i][indices[:, 0], indices[:, 1]] = True

    # Plot individual configs
    # for config in configs:
    #     plot_solution([config])

    final_configs = apply_all_transformations(configs)
    print(f"{len(final_configs)} possible configurations (including transformations).")

    visited_nodes = set()
    num_operations = 0

    results_frame = pd.DataFrame(columns=["Method", "Memoize Depth", "Mean Time", "Min Time", "Max Time",
                                          "Num Operations"])
    num_repetitions = 5

    for search_method in ["Iterative", "Recursive"]:
        for memoize_depth_selection in range(5):
            timing_results = np.zeros(num_repetitions)
            for rep_idx in range(num_repetitions):
                visited_nodes = set()
                num_operations = 0

                start_time = time.time()

                if search_method == "Iterative":
                    solution_configs = check_combs_iterative_brute_force(final_configs,
                                                                         memoize_depth=memoize_depth_selection)
                else:
                    solution_configs = check_combs_recursive_brute_force(final_configs, house_config,
                                                                         memoize_depth=memoize_depth_selection)
                finish_time = time.time()

                elapsed_time = finish_time - start_time
                timing_results[rep_idx] = elapsed_time

                if solution_configs:
                    print("solution found")
                else:
                    print("solution not found")

            results_frame.loc[len(results_frame)] = [search_method, memoize_depth_selection, timing_results.mean(),
                                                     timing_results.min(), timing_results.max(), num_operations]

    print(results_frame.to_markdown(index=False))

    # PLot final solution
    # plot_solution(solution_configs)
