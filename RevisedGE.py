import os
import time
import math
import argparse
import datetime

import numpy as np
import scipy.sparse as sparse
import osqp
import heapq
from numpy.linalg import norm
from qpsolvers import solve_qp


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename", type=str, help="Input filename")
    parser.add_argument("output_filename", type=str, help="Output filename")
    parser.add_argument("-i", "--init_size", dest='initial_size', type=int, help="The initial size of output convexhull (-1 means #nodes / 20)", default=-1)
    parser.add_argument("-s", "--size", dest='convexhull_size', type=int, help="The max size of output convexhull (-1 means #nodes / 10)", default=-1)
    parser.add_argument("-c", "--convergence", dest='convergence_distance', type=float, help="The criteria of convergence", default=0.06)
    parser.add_argument("-e", "--error", dest='tolerated_qp_error', type=float, help="How much distance error in QP solver is tolerated", default=0.03)
    parser.add_argument("-cr", "--convergence-rate", dest='convergence_change_rate', type=float, help="The criteria of convergence for KCHA", default=0.01)
    parser.add_argument("-si", "--sigma", dest='sigma', type=float, help="Sigma, the parameter of Gaussion kernel", default=0.3)
    args = parser.parse_args()
    return args


def load_data(input_filename):
    if os.path.splitext(args.input_filename)[1] == '.txt':
        with open(args.input_filename) as fp:
            node_count, dimension = [int(i) for i in fp.readline().split()]
            node_vectors = np.zeros((node_count, dimension))
            for line in fp:
                line = line.split()
                node_vectors[int(line[0])] = [float(i) for i in line[1:]]
    elif os.path.splitext(args.input_filename)[1] == '.npy':
        node_vectors = np.load(open(args.input_filename, 'rb'))
    else:
        raise Exception('input filename not end with txt or npy')
    return node_vectors


def init_convexhull(node_vectors):
    N, D = node_vectors.shape
    X = np.hstack((node_vectors, np.ones((N, 1)) * math.sqrt(D)))  # N * (D + 1)
    C = np.ones((N, N)) / N  # N * N
    X_i_minus_X_j = X.reshape((-1, 1, D + 1)) - X.reshape((1, -1, D + 1))  # N * N * (D + 1)
    square_norm_X_i_minus_X_j = (X_i_minus_X_j * X_i_minus_X_j).sum(axis=2)  # N * N
    K = np.exp(-square_norm_X_i_minus_X_j / (args.sigma * args.sigma))  # N * N
    while True:
        C_last = C
        C = C_last * np.sqrt(K / K.dot(C_last).clip(min=1e-100))
        if np.abs((C - C_last)).sum() / np.abs(C_last).sum() <= args.convergence_change_rate:
            break
    C_diag = C[np.diag_indices_from(C)]
    convexhull_indexes = set(np.argsort(C_diag)[-args.initial_size:])
    convexhull_indexes.update(node_vectors.argmax(axis=0))
    convexhull_indexes.update(node_vectors.argmin(axis=0))

    return convexhull_indexes


def write_out_convex_hall(nodes, convexhull_indexes):
    if not os.path.exists(os.path.split(args.output_filename)[0]):
        os.makedirs(os.path.split(args.output_filename)[0])
    if os.path.splitext(args.output_filename)[1] == '.txt':
        with open(args.output_filename, 'w') as fp:
            fp.write('%d\t%d\n' % (len(convexhull_indexes), nodes.shape[1]))
            for convex_hall_index in convexhull_indexes:
                fp.write('%d\t%s\n' % (convex_hall_index, '\t'.join(str(i) for i in nodes[convex_hall_index])))
    elif os.path.splitext(args.output_filename)[1] == '.npy':
        np.save(args.output_filename, nodes[convexhull_indexes])
    else:
        raise Exception('output filename not end with txt or npy')


def distance_point_to_convexhull_all_vector(z, X):
    S, d = X.shape
    global P, q, A

    # minimize (1/2)x^TPx + q^Tx; subject to Gx <= h & Ax = b
    P = sparse.csc_matrix(X.dot(X.T))
    q = -(X * z).sum(axis=1)
    G = sparse.csc_matrix(-sparse.eye(S))
    h = np.zeros(S)
    A = np.ones(S)
    b = 1
    a = solve_qp(P, q, G, h, A, b, solver='osqp')

    distance = z - a.dot(X)
    distance = math.sqrt(distance.dot(distance))
    return distance


def get_distance(row_node, convexhull, node_vectors):
    convexhull_list = list(convexhull)
    X = node_vectors[convexhull_list]  # S*d
    z = node_vectors[row_node]  # d
    return distance_point_to_convexhull_all_vector(z, X)


def find_next_node_to_convexhull(remaining_nodes, convexhull, node_vectors):
    global heap, distance, row, col
    remaining_nodes = list(remaining_nodes)
    heap = [(0, i, -1) for i in range(len(remaining_nodes))]
    distance_matrix = [[] for i in range(len(remaining_nodes))]
    while True:
        last_distance, col, row = heapq.heappop(heap)
        if row == len(remaining_nodes) - 1:
            break
        convexhull.add(remaining_nodes[col])
        if col == row + 1:
            distance = 0
        else:
            distance = get_distance(remaining_nodes[row + 1], convexhull, node_vectors)
        convexhull.remove(remaining_nodes[col])
        distance_matrix[col].append(distance)
        heapq.heappush(heap, (max(last_distance, distance), col, row + 1))
    nodes_to_be_removed = {remaining_nodes[idx] for idx, dis in enumerate(distance_matrix[col]) if abs(dis) <= args.tolerated_qp_error}
    return remaining_nodes[col], last_distance, nodes_to_be_removed


def clean_remaining_nodes(remaining_nodes, convexhull, node_vectors):
    for node in list(remaining_nodes):
        distance = get_distance(node, convexhull, node_vectors)
        if abs(distance) <= args.tolerated_qp_error:
            remaining_nodes.remove(node)


def clean_convexhull(convexhull, node_vectors):
    for node in list(convexhull):
        convexhull.remove(node)
        distance = get_distance(node, convexhull, node_vectors)
        if abs(distance) >= args.tolerated_qp_error:
            convexhull.add(node)


def get_convexhull(node_vectors):
    convexhull = init_convexhull(node_vectors)
    remaining_nodes = set(range(node_vectors.shape[0])) - convexhull
    clean_convexhull(convexhull, node_vectors)
    clean_remaining_nodes(remaining_nodes, convexhull, node_vectors)
    print('Initially %d nodes in convexhull, and %d nodes remains' % (len(convexhull), len(remaining_nodes)))
    while len(convexhull) <= args.convexhull_size:
        if len(remaining_nodes) == 0:
            print('End with no remaining nodes')
            break
        node, distance, nodes_in_convexhull = find_next_node_to_convexhull(remaining_nodes, convexhull, node_vectors)
        remaining_nodes -= set(nodes_in_convexhull)
        convexhull.add(node)
        clean_convexhull(convexhull, node_vectors)
        if distance <= args.convergence_distance:
            print('End with distance reaching threashold', distance)
            break
        print('%d nodes in convexhull, and %d nodes remains, and the farest distance is %f' % (len(convexhull), len(remaining_nodes), distance))
    else:
        print('End with reaching the setted max convexhull size, %d nodes remains' % len(remaining_nodes))
    return convexhull


def main():
    nodes = load_data(args.input_filename)
    if args.initial_size == -1:
        args.initial_size = len(nodes) // 20
    if args.convexhull_size == -1:
        args.convexhull_size = len(nodes) // 10
    convexhull_indexes = get_convexhull(nodes)
    write_out_convex_hall(nodes, convexhull_indexes)


if __name__ == '__main__':
    args = parse_argument()
    main()
