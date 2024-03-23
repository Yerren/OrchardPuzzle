import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

GRID_SIZE = 8


def check_house_overlap(map_in):
    return np.all(map_in[6, 3] == 1 or map_in[6, 4] == 1 or map_in[7, 3] == 1 or map_in[7, 4] == 1)


def check_overlap(map_1, map_2):
    return np.any((map_1 & map_2) != 0)


def check_if_config_valid(config_in, list_in):
    if check_house_overlap(config_in):
        return True
    for existing_config in list_in:
        if (config_in == existing_config).all():
            return True

    return False


configs = np.zeros((5, 8, 8), dtype='bool')
config_indices = np.array([[[7, 0], [5, 1], [0, 2], [3, 2], [5, 2], [6, 2], [1, 3], [5, 4], [4, 6], [5, 7]],
                           [[0, 0], [0, 3], [0, 5], [0, 6], [1, 4], [2, 2], [3, 0], [4, 1], [5, 0], [6, 0]],
                           [[0, 1], [1, 3], [2, 5], [3, 1], [3, 2], [3, 4], [3, 7], [5, 1], [6, 1], [7, 0]],
                           [[0, 3], [2, 2], [2, 4], [3, 3], [4, 1], [4, 2], [4, 4], [4, 5], [6, 0], [6, 6]],
                           [[0, 0], [3, 0], [3, 3], [4, 2], [5, 0], [5, 1], [5, 4], [5, 5], [6, 0], [6, 6]]])

for i, indices in enumerate(config_indices):
    configs[i][indices[:, 0], indices[:, 1]] = True

# Populate configs with unique transformations: translate 7x7s, flip LR, flip UP, rotate.
# This could be made more efficient by checking hashes, but probably not worth optimizing.

# Translate 7x7s
translated_configs = []
for config in configs:
    if not check_if_config_valid(config, translated_configs):
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

            if not check_if_config_valid(shifted_config, translated_configs):
                translated_configs.append(shifted_config)

# Flips (don't need to do lr+ud, as that is the same as 180 degree rotation.
flipped_configs = translated_configs.copy()
for config in translated_configs:
    lr_flip = np.fliplr(config)
    if not check_if_config_valid(lr_flip, flipped_configs):
        flipped_configs.append(lr_flip)

    ud_flip = np.flipud(config)
    if not check_if_config_valid(ud_flip, flipped_configs):
        flipped_configs.append(ud_flip)

# Rotations
rotated_configs = flipped_configs.copy()
for config in flipped_configs:
    rot90 = config
    for _ in range(3):
        rot90 = np.rot90(rot90)
        if not check_if_config_valid(rot90, rotated_configs):
            rotated_configs.append(rot90)

final_configs = np.array(rotated_configs)

# Fully naive brute force search
print(f"{len(final_configs)} configs")
print(f"{len(final_configs)**4} combinations to check")


def check_combs(configs_in):
    for choice_1 in configs_in:
        for choice_2 in configs_in:
            for choice_3 in configs_in:
                for choice_4 in configs_in:
                    output = choice_1 + choice_2 + choice_3 + choice_4
                    if check_house_overlap(output):
                        continue
                    if output.sum() == 40:
                        print("solution found:")
                        print(choice_1)
                        print(choice_2)
                        print(choice_3)
                        print(choice_4)

                        # Assuming you have four 8x8 boolean numpy arrays named array1, array2, array3, array4
                        # array1 = np.random.choice([True, False], size=(8, 8))
                        # array2 = np.random.choice([True, False], size=(8, 8))
                        # array3 = np.random.choice([True, False], size=(8, 8))
                        # array4 = np.random.choice([True, False], size=(8, 8))

                        fig, ax = plt.subplots()

                        # Create a custom colormap for each color
                        cmap_reds = plt.cm.Reds(np.arange(plt.cm.Reds.N))
                        cmap_blues = plt.cm.Blues(np.arange(plt.cm.Blues.N))
                        cmap_greens = plt.cm.Greens(np.arange(plt.cm.Greens.N))
                        cmap_purples = plt.cm.Purples(np.arange(plt.cm.Purples.N))
                        cmap_greys = plt.cm.Greys(np.arange(plt.cm.Greys.N))

                        # Set alpha (transparency) of the lowest intensity color to 0 (transparent)
                        cmap_reds[0, -1] = 0
                        cmap_blues[0, -1] = 0
                        cmap_greens[0, -1] = 0
                        cmap_purples[0, -1] = 0
                        cmap_greys[0, -1] = 0

                        # Create new colormaps
                        cmap_reds = ListedColormap(cmap_reds)
                        cmap_blues = ListedColormap(cmap_blues)
                        cmap_greens = ListedColormap(cmap_greens)
                        cmap_purples = ListedColormap(cmap_purples)
                        cmap_greys = ListedColormap(cmap_greys)

                        # Overlay the four arrays on top of each other in different colors
                        ax.imshow(choice_1, cmap=cmap_reds)
                        ax.imshow(choice_2, cmap=cmap_blues)
                        ax.imshow(choice_3, cmap=cmap_greens)
                        ax.imshow(choice_4, cmap=cmap_purples)

                        plt.show()

                        plt.show()
                        return True

check_combs(final_configs)

print("end")

