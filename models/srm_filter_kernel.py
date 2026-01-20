import torch

# 使用 helper 函数简化 tensor 创建
def t(data):
    return torch.tensor(data, dtype=torch.float32)

filter_class_1 = [
  t([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 0]
  ]),
  t([
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 0]
  ]),
  t([
    [0, 0, 1],
    [0, -1, 0],
    [0, 0, 0]
  ]),
  t([
    [0, 0, 0],
    [1, -1, 0],
    [0, 0, 0]
  ]),
  t([
    [0, 0, 0],
    [0, -1, 1],
    [0, 0, 0]
  ]),
  t([
    [0, 0, 0],
    [0, -1, 0],
    [1, 0, 0]
  ]),
  t([
    [0, 0, 0],
    [0, -1, 0],
    [0, 1, 0]
  ]),
  t([
    [0, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
  ])
]


filter_class_2 = [
  t([
    [1, 0, 0],
    [0, -2, 0],
    [0, 0, 1]
  ]),
  t([
    [0, 1, 0],
    [0, -2, 0],
    [0, 1, 0]
  ]),
  t([
    [0, 0, 1],
    [0, -2, 0],
    [1, 0, 0]
  ]),
  t([
    [0, 0, 0],
    [1, -2, 1],
    [0, 0, 0]
  ]),
]


filter_class_3 = [
  t([
    [-1, 0, 0, 0, 0],
    [0, 3, 0, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0]
  ]),
  t([
    [0, 0, -1, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
  ]),
  t([
    [0, 0, 0, 0, -1],
    [0, 0, 0, 3, 0],
    [0, 0, -3, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ]),
  t([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, -3, 3, -1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ]),
  t([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, -1]
  ]),
  t([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -1, 0, 0]
  ]),
  t([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, -3, 0, 0],
    [0, 3, 0, 0, 0],
    [-1, 0, 0, 0, 0]
  ]),
  t([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-1, 3, -3, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ])
]


filter_edge_3x3 = [
  t([
    [-1, 2, -1],
    [2, -4, 2],
    [0, 0, 0]
  ]),
  t([
    [0, 2, -1],
    [0, -4, 2],
    [0, 2, -1]
  ]),
  t([
    [0, 0, 0],
    [2, -4, 2],
    [-1, 2, -1]
  ]),
  t([
    [-1, 2, 0],
    [2, -4, 0],
    [-1, 2, 0]
  ]),
]

filter_edge_5x5 = [
  t([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ]),
  t([
    [0, 0, -2, 2, -1],
    [0, 0, 8, -6, 2],
    [0, 0, -12, 8, -2],
    [0, 0, 8, -6, 2],
    [0, 0, -2, 2, -1]
  ]),
  t([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1]
  ]),
  t([
    [-1, 2, -2, 0, 0],
    [2, -6, 8, 0, 0],
    [-2, 8, -12, 0, 0],
    [2, -6, 8, 0, 0],
    [-1, 2, -2, 0, 0]
  ]),
]

square_3x3 = t([
  [-1, 2, -1],
  [2, -4, 2],
  [-1, 2, -1]
])

square_5x5 = t([
  [-1, 2, -2, 2, -1],
  [2, -6, 8, -6, 2],
  [-2, 8, -12, 8, -2],
  [2, -6, 8, -6, 2],
  [-1, 2, -2, 2, -1]
])


all_hpf_list = filter_class_1 + filter_class_2 + filter_class_3 + filter_edge_3x3 + filter_edge_5x5 + [square_3x3, square_5x5]

hpf_3x3_list = filter_class_1 + filter_class_2 + filter_edge_3x3 + [square_3x3]
hpf_5x5_list = filter_class_3 + filter_edge_5x5 + [square_5x5]

normalized_filter_class_2 = [hpf / 2 for hpf in filter_class_2]
normalized_filter_class_3 = [hpf / 3 for hpf in filter_class_3]
normalized_filter_edge_3x3 = [hpf / 4 for hpf in filter_edge_3x3]
normalized_square_3x3 = square_3x3 / 4
normalized_filter_edge_5x5 = [hpf / 12 for hpf in filter_edge_5x5]
normalized_square_5x5 = square_5x5 / 12

all_normalized_hpf_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_class_3 + \
 normalized_filter_edge_3x3 + normalized_filter_edge_5x5 + [normalized_square_3x3, normalized_square_5x5]

normalized_hpf_3x3_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_edge_3x3 + [normalized_square_3x3]
normalized_hpf_5x5_list = normalized_filter_class_3 + normalized_filter_edge_5x5 + [normalized_square_5x5]

normalized_3x3_list = normalized_filter_edge_3x3 + [normalized_square_3x3]
normalized_5x5_list = normalized_filter_edge_5x5 + [normalized_square_5x5]
