# Ambulance Greedy Optimizer

This project implements a greedy algorithm to determine the optimal locations for ambulances in a given area. The goal is to minimize response time and maximize coverage using a data-driven approach.

## Project Structure

```
ambulance-greedy-optimizer
├── src
│   ├── app.py                # Entry point of the application
│   ├── algorithms
│   │   └── greedy.py        # Implementation of the greedy algorithm
│   └── types
│       └── __init__.py      # Custom types and data structures
├── tests
│   └── test_greedy.py       # Unit tests for the GreedyOptimizer class
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # List of required packages
└── README.md                # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd ambulance-greedy-optimizer
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```bash
python src/app.py
```

This will initialize the application, load the necessary data, and compute the optimal ambulance locations using the greedy algorithm.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.