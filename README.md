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
   * **For a more detailed explanation, read the excellent [solution document](https://docs.google.com/document/d/1DaRbQx-8kFIkgmmHDGZjvzpMoFwI1Id2qWXb3GmXIh0/edit#heading=h.9durt8hvhzzb) by Monotof1, for their [modified version of the puzzle](https://www.reddit.com/r/puzzles/comments/6xd9o4/the_orchard_challenge/) (which is arguably more interesting[^2])**
2. TODO

[^1]: Solving this algorithmically would be an interesting future extension.
[^2]: The removal of the "house" makes for a cleaner premise, while enforcing the use of different configurations leads to a more interesting solution.
