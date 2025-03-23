#!/usr/bin/env python
# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import itertools
import click
import pandas as pd
from neal import SimulatedAnnealingSampler
import dimod


def parse_inputs(data_file, capacity):
    df = pd.read_csv(data_file, names=['cost', 'weight'])
    if not capacity:
        capacity = int(0.8 * sum(df['weight']))
        print("\nSetting weight capacity to 80% of total: {}".format(capacity))
    return df['cost'], df['weight'], capacity


def build_knapsack_bqm(costs, weights, max_weight, A=1000):
    n = len(costs)
    m = math.ceil(math.log2(max_weight + 1)) 
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    for i in range(n):
        var = f'x{i}'
        bqm.add_variable(var, -costs[i] + A * (weights[i] ** 2) - 2 * A * max_weight * weights[i])

    for j in range(m):
        var = f's{j}'
        coef = A * ((2 ** j) ** 2) - 2 * A * max_weight * (2 ** j)
        bqm.add_variable(var, coef)

    for i in range(n):
        for k in range(i + 1, n):
            bqm.add_interaction(f'x{i}', f'x{k}', 2 * A * weights[i] * weights[k])

    for i in range(n):
        for j in range(m):
            bqm.add_interaction(f'x{i}', f's{j}', 2 * A * weights[i] * (2 ** j))

    for j in range(m):
        for l in range(j + 1, m):
            bqm.add_interaction(f's{j}', f's{l}', 2 * A * (2 ** j) * (2 ** l))

    bqm.offset += A * (max_weight ** 2)

    return bqm


def parse_solution(sampleset, n):
    best = sampleset.first.sample
    selected_item_indices = [i for i in range(n) if best.get(f'x{i}', 0) == 1]
    return selected_item_indices


def datafile_help(max_files=5):
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        datafiles = os.listdir(data_dir)
        help_text = "\nName of data file (under the 'data/' folder) to run on.\nOne of:\n"
        for file in datafiles[:max_files]:
            _, weights, _ = parse_inputs(os.path.join(data_dir, file), 1234)
            help_text += f"{file:20} {sum(weights)}\n"
        help_text += "\nDefault is to run on data/large.csv."
    except Exception:
        help_text = "\nName of data file (under the 'data/' folder) to run on.\nDefault is to run on data/large.csv."
    return help_text


filename_help = datafile_help()


@click.command()
@click.option('--filename', type=click.File(), default='data/large.csv',
              help=filename_help)
@click.option('--capacity', default=None,
              help="Maximum weight for the container. By default sets to 80% of the total.")
def main(filename, capacity):
    sampler = SimulatedAnnealingSampler()
    costs, weights, capacity = parse_inputs(filename, capacity)

    print("Building BQM for knapsack problem with {} items.".format(len(costs)))
    bqm = build_knapsack_bqm(costs, weights, capacity, A=1000)

    print("Submitting BQM to solver {}.".format(sampler.__class__.__name__))
    sampleset = sampler.sample(bqm, num_reads=1000, label='Example - Knapsack')

    selected = parse_solution(sampleset, len(costs))

    print("\nFound best solution with energy {}.".format(sampleset.first.energy))
    print("Selected item indices (0-indexed):", selected)


if __name__ == '__main__':
    main()
