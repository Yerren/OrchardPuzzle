# Overview
The [Dr Wood Orchard puzzle](https://dr-wood.fandom.com/wiki/Orchard) is a challenge in where you need to place **four varieties of trees**, each with **ten trees**, in an 8x8 grid such that each variety makes **five lines** of **four trees**. 
There is also a 2x2 "house" in the bottom centre of the grid, where no trees can be placed.

_This repo contains spoilers for the Dr Wood Orchard puzzle. Continue reading at your own risk._

# Method
1. We determine all possible configurations of one variety of tree such that the criteria of five lines of four trees are met. We solve this by hand, using the following key observations:
   * If every tree was used once, we would need 20 trees total (five lines of four trees). This implies each tree needs to contribute to two lines on average.
   * A single tree cannot contribute to three lines at once. If it were to do so, TODO
