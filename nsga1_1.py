#!/usr/bin/python3
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
import gutils

n_v = 15
snr_min, tp_min, sf_min = -5.5, 2.0, 7.0
xl_snr = [snr_min for _ in range(n_v)]
xl_tp = [tp_min for _ in range(n_v)]
xl_sf = [sf_min for _ in range(n_v)]
xl = np.concatenate((xl_snr, xl_tp, xl_sf))

snr_max, tp_max, sf_max = 27.8, 14.0, 12.0
xu_snr = [snr_max for _ in range(n_v)]
xu_tp = [tp_max for _ in range(n_v)]
xu_sf = [sf_max for _ in range(n_v)]
xu = np.concatenate((xu_snr, xu_tp, xu_sf))

min_d = 1
n_var = n_v * 3
n_obj = 2

def generate_vertexes(min_val, max_val, min_dist):
  coords = np.zeros(15)
  
  coords[0] = min_val
  coords[14] = max_val

  flag = True
  while flag or coords[2] - coords[0] < min_dist:
    coords[1:3] = np.random.uniform(coords[0], max_val, 2)
    coords[0:3].sort()
    flag = False

  flag = True
  while flag or coords[5] - coords[3] < min_dist:
    coords[3] = np.random.uniform(coords[0], coords[2])
    coords[4:6] = np.random.uniform(coords[3], max_val, 2)
    coords[3:6].sort()
    flag = False
  
  flag = True
  while flag or coords[8] - coords[6] < min_dist:
    coords[6] = np.random.uniform(coords[0], coords[5])
    coords[7:9] = np.random.uniform(coords[6], max_val, 2)
    coords[6:9].sort()
    flag = False
  
  flag = True
  while flag or coords[11] - coords[9] < min_dist:
    coords[9] = np.random.uniform(coords[0], coords[8])
    coords[10:12] = np.random.uniform(coords[9], max_val, 2)
    coords[9:12].sort()
    flag = False
  
  flag = True
  while flag or coords[14] - coords[12] < min_dist:
    coords[12] = np.random.uniform(coords[0], coords[11])
    coords[13] = np.random.uniform(coords[12], max_val)
    coords[12:15].sort()
    flag = False
  
  coords[3:12].sort()
  if coords[5] <= coords[2]:
    coords[5] = coords[2] + np.random.uniform(0.05, 0.25)
    coords[5] = min(max_val, coords[5])
  
  if coords[8] <= coords[5]:
    coords[8] = coords[5] + np.random.uniform(0.05, 0.25)
    coords[8] = min(max_val, coords[8])
  
  if coords[11] <= coords[8]:
    coords[11] = coords[8] + np.random.uniform(0.05, 0.25)
    coords[11] = min(max_val, coords[11])

  return coords

def generate_vertexes02(min_val, max_val, min_dist):
  coords = np.zeros(18)
  
  coords[0] = min_val
  coords[17] = max_val

  flag = True
  while flag or coords[2] - coords[0] < min_dist:
    coords[1:3] = np.random.uniform(coords[0], max_val, 2)
    coords[0:3].sort()
    flag = False

  flag = True
  while flag or coords[5] - coords[3] < min_dist:
    coords[3] = np.random.uniform(coords[0], coords[2])
    coords[4:6] = np.random.uniform(coords[3], max_val, 2)
    coords[3:6].sort()
    flag = False
  
  flag = True
  while flag or coords[8] - coords[6] < min_dist:
    coords[6] = np.random.uniform(coords[0], coords[5])
    coords[7:9] = np.random.uniform(coords[6], max_val, 2)
    coords[6:9].sort()
    flag = False
  
  flag = True
  while flag or coords[11] - coords[9] < min_dist:
    coords[9] = np.random.uniform(coords[0], coords[8])
    coords[10:12] = np.random.uniform(coords[9], max_val, 2)
    coords[9:12].sort()
    flag = False
  
  flag = True
  while flag or coords[14] - coords[12] < min_dist:
    coords[12] = np.random.uniform(coords[0], coords[11])
    coords[13:15] = np.random.uniform(coords[12], max_val, 2)
    coords[12:15].sort()
    flag = False
  
  flag = True
  while flag or coords[17] - coords[15] < min_dist:
    coords[15] = np.random.uniform(coords[0], coords[14])
    coords[16] = np.random.uniform(coords[12], max_val)
    coords[15:17].sort()
    flag = False

  # 3 - 5, 6 - 8, 9 - 11, 12 - 14, 15 - 17
  coords[3:14].sort()
  if coords[5] <= coords[2]:
    coords[5] = coords[2] + np.random.uniform(0.05, 0.25)
    coords[5] = min(max_val, coords[5])
  
  if coords[8] <= coords[5]:
    coords[8] = coords[5] + np.random.uniform(0.05, 0.25)
    coords[8] = min(max_val, coords[8])
  
  if coords[11] <= coords[8]:
    coords[11] = coords[8] + np.random.uniform(0.05, 0.25)
    coords[11] = min(max_val, coords[11])
  
  if coords[14] <= coords[11]:
    coords[14] = coords[11] + np.random.uniform(0.05, 0.25)
    coords[14] = min(max_val, coords[14])

  return coords

def swap(a, b):
  aux = a
  a = b
  b = aux
  return a, b

def check_v(v, min_val, max_val, min_dist):
  v[0] = min_val
  v[11] = max_val

  v[0:3].sort()
  while v[2] - v[0] < min_dist:
    v[2] += np.random.uniform(0.05, 0.25)
    v[2] = min(max_val, v[2])
    v[0:3].sort()
  
  v[3:6].sort()
  while v[5] - v[3] < min_dist:
    v[5] += np.random.uniform(0.05, 0.25)
    v[5] = min(max_val, v[5])
    v[3] -= np.random.uniform(0.05, 0.25)
    v[3] = max(min_val, v[3])
    v[3:6].sort()

  v[6:9].sort()
  while v[8] - v[6] < min_dist:
    v[8] += np.random.uniform(0.05, 0.25)
    v[8] = min(max_val, v[8])
    v[6] -= np.random.uniform(0.05, 0.25)
    v[6] = max(min_val, v[6])
    v[6:9].sort()

  v[9:12].sort()
  while v[11] - v[9] < min_dist:
    v[11] += np.random.uniform(0.05, 0.25)
    v[11] = min(max_val, v[11])
    v[9] -= np.random.uniform(0.05, 0.25)
    v[9] = max(min_val, v[9])
    v[9:12].sort()
  
  v[12:15].sort()
  while v[14] - v[12] < min_dist:
    v[12] -= np.random.uniform(0.05, 0.25)
    v[12] = max(min_val, v[12]) 
    v[12:15].sort()
  
  if v[3] > v[2]:
    v[3], v[2] = swap(v[3], v[2])
  
  if v[6] > v[5]:
    v[6], v[5] = swap(v[6], v[5])
  
  if v[9] > v[8]:
    v[9], v[8] = swap(v[9], v[8])
  
  if v[12] > v[11]:
    v[12], v[11] = swap(v[12], v[11])
  
  if v[5] <= v[2]:
    v[5] = v[2] + np.random.uniform(0.05, 0.25)
    v[5] = min(max_val, v[5])
  
  if v[8] <= v[5]:
    v[8] = v[5] + np.random.uniform(0.05, 0.25)
    v[8] = min(max_val, v[8])
  
  if v[11] <= v[8]:
    v[11] = v[8] + np.random.uniform(0.05, 0.25)
    v[11] = min(max_val, v[11])

class FuzzyProblem(Problem):
  def __init__(self):
    self.num_run = 0
    super().__init__(n_var=n_var,
                      n_obj=n_obj,
                      n_constr=0,
                      xl=xl,
                      xu=xu,
                      elementwise_evaluation=True)

  def _check_x(self, x):
    snr = x[0:15]
    tp = x[15:30]
    sf = x[30:45]

    check_v(snr, snr_min, snr_max, min_d)
    check_v(tp, tp_min, tp_max, min_d)
    check_v(sf, sf_min, sf_max, min_d)

  def _simulate(self, x):
    results = []
    for i in x:      
      gutils.fill_fll3(i)
      gutils.simulate_range()
      pdr, energy = gutils.calc_range()
      results.append(np.array([-pdr, energy]))

    return np.array(results)

  def _evaluate(self, x, out, *args, **kwargs):
    if self.num_run > 0:
      for i in x:
        self._check_x(i)
    
    out['F'] = [self._simulate(x)]
    
    self.num_run += 1

def main(pop_size=2, n_gen=1):
  X = []
  for i in range(pop_size):
    snr = generate_vertexes(snr_min, snr_max, min_d)
    tp = generate_vertexes(tp_min, tp_max, min_d)
    sf = generate_vertexes(sf_min, sf_max, min_d)
    X.append(np.concatenate((snr, tp, sf)))

  problem = FuzzyProblem()
  algorithm = NSGA2(pop_size=pop_size, 
                    mutation=PolynomialMutation(prob=0.2),
                    crossover=SinglePointCrossover(prob=1.0),
                    eliminate_duplicates=True,
                    sampling=np.array(X))

  res = minimize(problem,
                algorithm,
                ('n_gen', n_gen),
                save_history=True,
                verbose=True)

  dir_r = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-allinone/ns-3.38/scratch/genetic/results'

  print('Solutions found:')
  for fitness in res.F:
    print(fitness)
    with open(f'{dir_r}/solutions_nsga1{pop_size}-{n_gen}.csv', mode='a') as file:
      file.write(','.join(map(str, fitness))+'\n')

  solutions = res.pop.get('X')
  fitness = res.pop.get('F')
  with open(f'{dir_r}/results_nsga1{pop_size}-{n_gen}.csv', mode='a') as file:
    file.write(','.join(map(str, [f'x{i}' for i in range(1, len(solutions[i])+1)]))+','+'pdr,energy\n')
    for i in range(len(solutions)):
      file.write(','.join(map(str, solutions[i]))+','+','.join(map(str, fitness[i]))+'\n')

if __name__ == '__main__':
  #main(80, 90)
  #v = [10.77994548965534,12.348506259705703,14.0,7.0,10.51392789771179,10.741762367039113,7.7954471330686435,8.084589348165428,10.974291464817608,8.876966638890405,9.752189523902889,11.025812525676562,10.954208302282966,11.647803493210008,11.88410311927191,7.760614242304446,9.774751476724772,12.0]
  
  #gutils.simulate(321, 'ns3::AdrComponent')
  pdr, energy = gutils.calc(321, '', 'component')
  print(f'pdr={pdr} and energy={energy}')

