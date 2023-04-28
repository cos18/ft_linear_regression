from typing import List

def get_theta() -> List[int]:
  try:
    with open('ft_linear_regression/data/theta', 'r') as theta_file:
      return list(map(float, theta_file.readline().split()))
  except Exception as e:
    raise RuntimeError('Error when reading theta file!')
