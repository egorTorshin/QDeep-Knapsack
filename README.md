# Important note
This repository is a fork of D-Wave's original knapsack. The contributors to this fork do not claim ownership or authorship of the original codebase. All credit for the original work belongs to D-Wave Systems and its respective contributors.

# Knapsack

The [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) is a
well-known optimization problem. It is encountered, for example, in packing
shipping containers. A shipping container has a weight capacity which it can hold.
Given a collection of items to be shipped, where each item has a value and a
weight, the problem is to select the optimal items to pack in the shipping
container. This optimization problem can be defined as an objective with a constraint:

* **Objective:** Maximize freight value (sum of values of the selected items).
* **Constraint:** Total freight weight (sum of weights of the selected items) must
  be less than or equal to the container's capacity.

## Usage

To run the default demo, enter the command:

```bash
python knapsack.py
```

To view available options, enter the command:

```bash
python knapsack.py --help
```

Command-line arguments let you select one of several data sets (under the `/data`
folder) and set the freight capacity. The data files are formulated as rows of
items, each defined as a pair of weight and value.  


## License

Released under the Apache License 2.0. See [LICENSE](LICENSE) file.
