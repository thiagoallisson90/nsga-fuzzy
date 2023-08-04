#!/usr/bin/python3
# Based on <https://github.com/sahutkarsh/NSGA-II/blob/master/NSGA-II.ipynb>
import math
import random
import matplotlib.pyplot as plt
import numpy as np

def objective1(x):
  y = -x**3
  return y

def objective2(x):
  y = -(x-2)**2
  return y

population = 25
max_gen = 501
min_value= -100
max_value= 100
mut_rate = 0.25
cross_rate = 1

def index_locator(a,list):
  for i in range(0,len(list)):
    if list[i] == a:
      return i
  return -1

def sort_by_values(list1, values):
  sorted_list = []
  while(len(sorted_list)!=len(list1)):
    if index_locator(min(values),values) in list1:
      sorted_list.append(index_locator(min(values),values))
    values[index_locator(min(values),values)] = math.inf
  return sorted_list

def crowding_distance(values1, values2, front):
  distance = [0 for i in range(0,len(front))]
  sorted1 = sort_by_values(front, values1[:])
  sorted2 = sort_by_values(front, values2[:])
  distance[0] = 9999999999999999
  distance[len(front) - 1] = 9999999999999999
  for k in range(1,len(front)-1):
    distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
  for k in range(1,len(front)-1):
    distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
  return distance

def crossover(a,b):
  r=random.random()
  if r < cross_rate:
    return mutation((a+b)/2)
  else:
    return mutation((a-b)/2)
  
def mutation(solution):
  mutation_prob = random.random()
  if mutation_prob < mut_rate:
    solution = min_value+(max_value-min_value)*random.random()
  return solution

def non_dominated_sorting_algorithm(values1, values2):
  S=[[] for i in range(0,len(values1))]
  front = [[]]
  n=[0 for i in range(0,len(values1))]
  rank = [0 for i in range(0, len(values1))]

  for p in range(0,len(values1)):
      S[p]=[]
      n[p]=0
      for q in range(0, len(values1)):
          if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
              if q not in S[p]:
                  S[p].append(q)
          elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
              n[p] = n[p] + 1
      if n[p]==0:
          rank[p] = 0
          if p not in front[0]:
              front[0].append(p)
  i = 0
  while(front[i] != []):
      Q=[]
      for p in front[i]:
          for q in S[p]:
              n[q] =n[q] - 1
              if( n[q]==0):
                  rank[q]=i+1
                  if q not in Q:
                      Q.append(q)
      i = i+1
      front.append(Q)
  del front[len(front)-1]
  return front

def generate_solution(min_val, max_val, min_dist):
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


def nsga2(population,max_gen,min_value,max_value):  
  gen_no=0
  solution=[min_value+(max_value-min_value)*random.random() for i in range(0,population)]
    
  while(gen_no<max_gen):
    objective1_values = [objective1(solution[i])for i in range(0,population)]
    objective2_values = [objective2(solution[i])for i in range(0,population)]
    non_dominated_sorted_solution = non_dominated_sorting_algorithm(objective1_values[:],objective2_values[:])
    print('Best Front for Generation:',gen_no)
    for values in non_dominated_sorted_solution[0]:
        print(round(solution[values],3),end=" ")
    print("\n")
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(objective1_values[:],objective2_values[:],non_dominated_sorted_solution[i][:]))
    solution2 = solution[:]
    
    while(len(solution2)!=2*population):
        a1 = random.randint(0,population-1)
        b1 = random.randint(0,population-1)
        solution2.append(crossover(solution[a1],solution[b1]))
    objective1_values2 = [objective1(solution2[i])for i in range(0,2*population)]
    objective2_values2 = [objective2(solution2[i])for i in range(0,2*population)]
    non_dominated_sorted_solution2 = non_dominated_sorting_algorithm(objective1_values2[:],objective2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(objective1_values2[:],objective2_values2[:],non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_locator(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==population):
                break
        if (len(new_solution) == population):
            break
    solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1
  
  return [objective1_values, objective2_values]

def non_dominating_curve_plotter(objective1_values, objective2_values):
  plt.figure(figsize=(15,8))
  objective1 = [i * -1 for i in objective1_values]
  objective2 = [j * -1 for j in objective2_values]
  plt.xlabel('Objective Function 1', fontsize=15)
  plt.ylabel('Objective Function 2', fontsize=15)
  plt.scatter(objective1, objective2, c='red')
  plt.show()
  
objective1_values, objective2_values = nsga2(population,max_gen,min_value,max_value)

non_dominating_curve_plotter(objective1_values, objective2_values)
