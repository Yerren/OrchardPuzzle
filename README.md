# Overview
The [Dr Wood Orchard puzzle](https://dr-wood.fandom.com/wiki/Orchard) is a challenge in where you need to place **four varieties of trees**, each with **ten trees**, in an 8x8 grid such that each variety makes **five lines** of **four trees**. 
There is also a 2x2 "house" in the bottom centre of the grid, where no trees can be placed.

_This repo contains spoilers for the Dr Wood Orchard puzzle. Continue at the risk of spoiling the puzzle for yourself._

# Method
1. We determine all possible configurations of one variety of tree such that the criteria of five lines of four trees are met. We solve this by hand[^1], using the following key observations:
   * If every tree was used once, we would need 20 trees total (five lines of four trees). This implies each tree needs to contribute to two lines on average.
   * A single tree cannot contribute to three (or more) lines at once.
     * Consider three lines (of four trees) intersecting at one tree.
     * This would require all ten trees for just those three lines.
     * There is no way that any more lines of four trees could be formed in this configuration, as that would require at least one of the three existing lines to contribute more than one tree (which is impossible).
   * From the two points above, we can conclude that every tree must contribute to exactly two lines.
     * Therefore, we also know that each line must intersect with every other line.
     * Additionally, no two lines can be parallel (as then they wouldn't intersect).
   * There are eight possible gradients of line that can fit on the 8x8 grid.
     * We can now go through and check all selections of five lines can form a valid configuration.
     * This results in five valid configurations (excluding rotations, reflections, and translations).
       * <img src="https://github.com/Yerren/OrchardPuzzle/blob/main/config_1.png?raw=true" height="300" /> 
       * <img src="https://github.com/Yerren/OrchardPuzzle/blob/main/config_2.png?raw=true" height="300" /> 
       * <img src="https://github.com/Yerren/OrchardPuzzle/blob/main/config_3.png?raw=true" height="300" /> 
       * <img src="https://github.com/Yerren/OrchardPuzzle/blob/main/config_4.png?raw=true" height="300" /> 
       * <img src="https://github.com/Yerren/OrchardPuzzle/blob/main/config_5.png?raw=true" height="300" /> 
   * **For a more detailed explanation, read the excellent [solution document](https://docs.google.com/document/d/1DaRbQx-8kFIkgmmHDGZjvzpMoFwI1Id2qWXb3GmXIh0/edit#heading=h.9durt8hvhzzb) by Monotof1, for their [modified version of the puzzle](https://www.reddit.com/r/puzzles/comments/6xd9o4/the_orchard_challenge/) (which is arguably more interesting[^2])**
2. We apply transformations to the valid configurations:
  * For each new configuration, we check that they are still valid, and that they haven't already been found (as well as that their left-right mirror hasn't already been found, as that is the only symmetry, due to the "house").
  * Translation: shift the 7x7 configurations by one step right, down, and right+down.
  * Flip: flip all configurations found so far horizontally and vertically.
  * Rotate: rotate all configurations found so far 90, 180, and 270 degrees.
  * This leaves us with 40 total valid configurations.
3. We search for valid (non-overlapping) selections of four configurations:
  * There are many ways to do this, but I experimented with two basic approaches:
      1. Brute force iterative search:
           * Iteratively select configurations, at each stage checking whether they are valid (no overlaps).
           * If four configurations have been selected, without overlaps, a solution is found.
      2. Brute force recursive search:
           * Very similar to the iterative version, just implemented recursively.
   * For both iterative and recursive search, I implemented some simple memoization.
       * Before checking a selection of configurations, we check whether the current board state (or its left-right mirror) has been seen before.
       * If it has been seen, do not continue searching deeper.
       * If it has not been seen, add it to the hash table (set) of seen configurations.
   * These approaches find the following valid solution:
     *  <img src="https://github.com/Yerren/OrchardPuzzle/blob/main/solution_plot.png?raw=true" height="300" /> 

# Experiments
Out of curiosity, I evaluated the (time) efficiency for the iterative vs recursive searches. I also evaluated the impact of memoization.
A few key considerations:
* The search space is relatively small and shallow, so these experiments are a bit trivial. However, it would be an interesting future avenue of exploration to create similar puzzles at larger scales (e.g., larger grid sizes, more trees, and more trees required per line).
* To help account for some of the variability when timing, I repeat each search multiple times and take the average of their durations.
* The operation to check whether a selection is valid is very cheap, particularly compared to the memoization operations. Therefore, it is also interesting to consider the number of "operations" that are performed.
    * Here I (somewhat arbitrarily) consider an "operation" to be either checking whether a selection is valid, or checking if it has already been seen (and adding it to the hash table if not).
    * Because of the relatively large cost of the memoization operation, it is also worth considering only performing them at lower levels of the search (i.e., only checking whether selections of one or two configurations have been seen before).
        * Therefore, I also checked the performance of only performing memoization up to certain depths (one to four selected configurations) of the search.

| Method    |   Memoize Depth |   Mean Time |   Min Time |   Max Time |   Num Operations |
|:----------|----------------:|------------:|-----------:|-----------:|-----------------:|
| Iterative |               0 |   0.0263247 |  0.0260231 |  0.027025  |             9444 |
| Iterative |               1 |   0.0247277 |  0.0240231 |  0.0250237 |             7973 |
| Iterative |               2 |   0.0698653 |  0.0690632 |  0.0720675 |             6512 |
| Iterative |               3 |   0.461251  |  0.457474  |  0.470981  |             6573 |
| Iterative |               4 |   0.652523  |  0.649108  |  0.660623  |             7650 |
| Recursive |               0 |   0.0212203 |  0.0210187 |  0.0220215 |             9445 |
| Recursive |               1 |   0.0200196 |  0.0195215 |  0.0210187 |             7975 |
| Recursive |               2 |   0.0709893 |  0.0700641 |  0.0720677 |             6514 |
| Recursive |               3 |   0.465146  |  0.460429  |  0.468962  |             6575 |
| Recursive |               4 |   0.666838  |  0.655117  |  0.681166  |             7652 |

[^1]: Solving this algorithmically would be an interesting future extension.
[^2]: The removal of the "house" makes for a cleaner premise, while enforcing the use of different configurations leads to a more interesting solution.
