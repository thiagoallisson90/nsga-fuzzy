import concurrent.futures
import math
import numpy as np
import os
import pandas as pd

ns3_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-allinone/ns-3.38'
ns3 = f'{ns3_dir}/./ns3'
ag_dir = 'scratch/genetic'

# prefix={'F', ''}
def calc_pdr(adr, num_reps=4, prefix='F'):
	files = [f'{ns3_dir}/phyPerformance102{prefix}ADR-{i}.csv' for i in range(1, num_reps+1)]
	x_s = [pd.read_csv(file)['received'].mean() for file in files]
	adr['mean_pdr'] = np.mean(x_s)

# prefix={'F', ''}
def calc_pdr_range(adr, prefix='F', n_runs=[96, 239, 116, 134, 64, 287]):
	files = [f'{ns3_dir}/phyPerformance102{prefix}ADR-{i}.csv' for i in n_runs]
	x_s = [pd.read_csv(file)['received'].mean() for file in files]
	adr['mean_pdr'] = np.mean(x_s)

# prefix={'component', 'fuzzy'}
def calc_energy(adr, num_reps=4, prefix='fuzzy'):
	files = [f'{ns3_dir}/battery-level-{prefix}-{i}.txt' for i in range(1, num_reps+1)]
	x_s = [(10000 - pd.read_csv(file, names=['num', 'energy'])['energy'].mean()) for file in files]
	adr['mean_energy'] = np.mean(x_s)
	
# prefix={'component', 'fuzzy'}
def calc_energy_range(adr, prefix='fuzzy', n_runs=[96, 239, 116, 134, 64, 287]):
	files = [f'{ns3_dir}/battery-level-{prefix}-{i}.txt' for i in n_runs]
	x_s = [(10000 - pd.read_csv(file, names=['num', 'energy'])['energy'].mean()) for file in files]
	adr['mean_energy'] = np.mean(x_s)

# adr_type={'ns3::AdrComponent', 'ns3::AdrFuzzy'}
def execute(run, adr_type='ns3::AdrFuzzy'):
	cmd = f"{ns3} run \"scenario102 --adrType={adr_type} --nDevices=136 --intervalTx=15 --nRun={run}\""
	os.system(cmd)

def compile_now():
	cmd = f'{ns3} configure -d optimized --enable-tests --enable-examples --enable-mpi'
	os.system(cmd)
	os.system(ns3)

# adr_type={'ns3::AdrComponent', 'ns3::AdrFuzzy'}
def simulate(num_reps=4, adr_type='ns3::AdrFuzzy'):
	executor = concurrent.futures.ThreadPoolExecutor()
	
	num_threads, start, end = 10, 0, 0
	its = math.floor(num_reps / num_threads)
	if its > 0:
		for i in range(0, its):
			start = i * num_threads + 1
			end = start + num_threads if i < its-1 else start + (num_reps - start) + 1
			compile_now()
			results = [executor.submit(execute, i, adr_type) for i in range(start, end)]
			concurrent.futures.wait(results)
	else:
		compile_now()
		results = [executor.submit(execute, i, adr_type) for i in range(1, num_reps+1)]
		concurrent.futures.wait(results)

	executor.shutdown()

# adr_type={'ns3::AdrComponent', 'ns3::AdrFuzzy'}
def simulate_range(adr_type='ns3::AdrFuzzy', n_runs = [96, 239, 116, 134, 64, 287]):
	executor = concurrent.futures.ThreadPoolExecutor()
	compile_now()
	results = [executor.submit(execute, i, adr_type) for i in n_runs]
	concurrent.futures.wait(results)
	executor.shutdown()

def simulate_one(adr_type='ns3::AdrFuzzy', n_run=1):
	compile_now()
	execute(n_run, adr_type)

def calc_one(prefix_pdr='F', prefix_energy='fuzzy', n_run=1):
	executor = concurrent.futures.ThreadPoolExecutor()
	file = f'{ns3_dir}/phyPerformance102{prefix_pdr}ADR-{n_run}.csv'
	mean_pdr = pd.read_csv(file)['received'].mean()
	file = f'{ns3_dir}/battery-level-{prefix_energy}-{n_run}.txt'
	mean_energy = (10000 - pd.read_csv(file, names=['num', 'energy'])['energy'].mean())
	return mean_pdr, mean_energy	

def calc(num_reps=4, prefix_pdr='F', prefix_energy='fuzzy'):
	adr = {}
	executor = concurrent.futures.ThreadPoolExecutor()
	results = [executor.submit(calc_pdr, adr, num_reps, prefix_pdr), 
     executor.submit(calc_energy, adr, num_reps, prefix_energy)]
	concurrent.futures.wait(results)
	executor.shutdown()
	return adr['mean_pdr'], adr['mean_energy']

def calc_range(prefix_pdr='F', prefix_energy='fuzzy'):
	adr = {}
	executor = concurrent.futures.ThreadPoolExecutor()
	results = [executor.submit(calc_pdr_range, adr, prefix_pdr), 
     executor.submit(calc_energy_range, adr, prefix_energy)]
	concurrent.futures.wait(results)
	executor.shutdown()
	return adr['mean_pdr'], adr['mean_energy']

def fill_fll(vertexes, rules1, rules2, fll='fadr.fll'):	
	filename = \
		f'{ns3_dir}/src/lorawan/examples/{fll}'
	with open(filename, 'w') as file:
		file.write(\
			'Engine: FADR\n' +
			'\tdescription: An Engine for ADR LoRaWAN\n' +
			'InputVariable: snr\n'+
			'\tdescription: Signal Noise Ratio\n'
			'\tenabled: true\n'+
			'\trange: -6.0  30.0\n'+
			'\tlock-range: true\n'+
			f'\tterm: poor       Triangle {vertexes[0]} {vertexes[1]} {vertexes[2]}\n'+
			f'\tterm: acceptable Triangle {vertexes[3]} {vertexes[4]} {vertexes[5]}\n'+
			f'\tterm: good       Triangle {vertexes[6]} {vertexes[7]} {vertexes[8]}\n'+
			'OutputVariable: tp\n'+
			'\tdescription: TP based on Mamdani inference\n'+
			'\tenabled: true\n'+
			'\trange:   2  14\n'+
			'\tlock-range: false\n'+
			'\taggregation: Maximum\n'+
			'\tdefuzzifier: Centroid 10\n'+
			'\tdefault: nan\n'+
			'\tlock-previous: false\n'+
			f'\tterm:  low      Triangle  {vertexes[9]}  {vertexes[10]} {vertexes[11]}\n'+
			f'\tterm:  average  Triangle  {vertexes[12]} {vertexes[13]} {vertexes[14]}\n'+
			f'\tterm:  high     Triangle  {vertexes[15]} {vertexes[16]} {vertexes[17]}\n'+
			'OutputVariable: sf\n'+
			'\tdescription: SF based on Mamdani inference\n'+
			'\tenabled: true\n'+
			'\trange: 7  12\n'+
			'\tlock-range: false\n'+
			'\taggregation: Maximum\n'+
			'\tdefuzzifier: Centroid 10\n'+
			'\tdefault: nan\n'+
			'\tlock-previous: false\n'+
			f'\tterm:  low     Triangle  {vertexes[18]} {vertexes[19]} {vertexes[20]}\n'+
			f'\tterm:  average Triangle  {vertexes[21]} {vertexes[22]} {vertexes[23]}\n'+
			f'\tterm:  high    Triangle  {vertexes[24]} {vertexes[25]} {vertexes[26]}\n'+
			'RuleBlock: mamdani\n'+
			'\tdescription: Mamdani Inference for TP\n'+
			'\tenabled: true\n'+
			'\tconjunction: Minimum\n'+
			'\tdisjunction: Maximum\n'+
			'\timplication: Minimum\n'+
			'\tactivation: General\n'+
			f'\trule: if snr is poor then tp is {rules1[0]}\n'+
			f'\trule: if snr is acceptable then tp is {rules1[1]}\n'+
			f'\trule: if snr is good then tp is {rules1[2]}\n'+
			'RuleBlock: mamdani\n'+
			'\tdescription: Mamdani Inference for SF\n'+
			'\tenabled: true\n'+
			'\tconjunction: Minimum\n'+
			'\tdisjunction: Maximum\n'+
			'\timplication: Minimum\n'+
			'\tactivation: General\n'+
			f'\trule: if snr is poor then sf is {rules2[0]}\n'+
			f'\trule: if snr is acceptable then sf is {rules2[1]}\n'+
			f'\trule: if snr is good then sf is {rules2[2]}\n'
		)

def fill_fll1(vertexes, fll='fadr.fll'):	
	filename = \
		f'{ns3_dir}/src/lorawan/examples/{fll}'
	with open(filename, 'w') as file:
		file.write(\
			'Engine: FADR\n' +
			'\tdescription: An Engine for ADR LoRaWAN\n' +
			'InputVariable: snr\n'+
			'\tdescription: Signal Noise Ratio\n'
			'\tenabled: true\n'+
			'\trange: -6.0  30.0\n'+
			'\tlock-range: true\n'+
			f'\tterm: poor       Triangle {vertexes[0]} {vertexes[1]} {vertexes[2]}\n'+
			f'\tterm: acceptable Triangle {vertexes[3]} {vertexes[4]} {vertexes[5]}\n'+
			f'\tterm: good       Triangle {vertexes[6]} {vertexes[7]} {vertexes[8]}\n'+
			'OutputVariable: tp\n'+
			'\tdescription: TP based on Mamdani inference\n'+
			'\tenabled: true\n'+
			'\trange:   2  14\n'+
			'\tlock-range: false\n'+
			'\taggregation: Maximum\n'+
			'\tdefuzzifier: Centroid 10\n'+
			'\tdefault: nan\n'+
			'\tlock-previous: false\n'+
			f'\tterm:  low      Triangle  {vertexes[9]}  {vertexes[10]} {vertexes[11]}\n'+
			f'\tterm:  average  Triangle  {vertexes[12]} {vertexes[13]} {vertexes[14]}\n'+
			f'\tterm:  high     Triangle  {vertexes[15]} {vertexes[16]} {vertexes[17]}\n'+
			'OutputVariable: sf\n'+
			'\tdescription: SF based on Mamdani inference\n'+
			'\tenabled: true\n'+
			'\trange: 7  12\n'+
			'\tlock-range: false\n'+
			'\taggregation: Maximum\n'+
			'\tdefuzzifier: Centroid 10\n'+
			'\tdefault: nan\n'+
			'\tlock-previous: false\n'+
			f'\tterm:  low     Triangle  {vertexes[18]} {vertexes[19]} {vertexes[20]}\n'+
			f'\tterm:  average Triangle  {vertexes[21]} {vertexes[22]} {vertexes[23]}\n'+
			f'\tterm:  high    Triangle  {vertexes[24]} {vertexes[25]} {vertexes[26]}\n'+
			'RuleBlock: mamdani\n'+
			'\tdescription: Mamdani Inference for TP\n'+
			'\tenabled: true\n'+
			'\tconjunction: Minimum\n'+
			'\tdisjunction: Maximum\n'+
			'\timplication: Minimum\n'+
			'\tactivation: General\n'+
		f'\trule: if snr is poor then tp is high\n'+
			f'\trule: if snr is acceptable then tp is average\n'+
			f'\trule: if snr is good then tp is low\n'+
			'RuleBlock: mamdani\n'+
			'\tdescription: Mamdani Inference for SF\n'+
			'\tenabled: true\n'+
			'\tconjunction: Minimum\n'+
			'\tdisjunction: Maximum\n'+
			'\timplication: Minimum\n'+
			'\tactivation: General\n'+
			f'\trule: if snr is poor then sf is high\n'+
			f'\trule: if snr is acceptable then sf is average\n'+
			f'\trule: if snr is good then sf is low\n'
		)

def fill_fll2(vertexes, fll='fadr.fll'):	
	filename = \
		f'{ns3_dir}/src/lorawan/examples/{fll}'
	with open(filename, 'w') as file:
		file.write(\
			'Engine: FADR\n' +
			'\tdescription: An Engine for ADR LoRaWAN\n' +
			'InputVariable: snr\n'+
			'\tdescription: Signal Noise Ratio\n'
			'\tenabled: true\n'+
			'\trange: -6.0  30.0\n'+
			'\tlock-range: true\n'+
			f'\tterm: very_poor       Triangle {vertexes[0]} {vertexes[1]} {vertexes[2]}\n'+
			f'\tterm: poor            Triangle {vertexes[3]} {vertexes[4]} {vertexes[5]}\n'+
			f'\tterm: acceptable      Triangle {vertexes[6]} {vertexes[7]} {vertexes[8]}\n'+
			f'\tterm: good            Triangle {vertexes[9]} {vertexes[10]} {vertexes[11]}\n'+
			'OutputVariable: tp\n'+
			'\tdescription: TP based on Mamdani inference\n'+
			'\tenabled: true\n'+
			'\trange:   2  14\n'+
			'\tlock-range: false\n'+
			'\taggregation: Maximum\n'+
			'\tdefuzzifier: Centroid 10\n'+
			'\tdefault: nan\n'+
			'\tlock-previous: false\n'+
			f'\tterm:  very_low  Triangle  {vertexes[12]} {vertexes[13]} {vertexes[14]}\n'+
			f'\tterm:  low       Triangle  {vertexes[15]} {vertexes[16]} {vertexes[17]}\n'+
			f'\tterm:  average   Triangle  {vertexes[18]} {vertexes[19]} {vertexes[20]}\n'+
			f'\tterm:  high      Triangle  {vertexes[21]} {vertexes[22]} {vertexes[23]}\n'+
			'OutputVariable: sf\n'+
			'\tdescription: SF based on Mamdani inference\n'+
			'\tenabled: true\n'+
			'\trange: 7  12\n'+
			'\tlock-range: false\n'+
			'\taggregation: Maximum\n'+
			'\tdefuzzifier: Centroid 10\n'+
			'\tdefault: nan\n'+
			'\tlock-previous: false\n'+
			f'\tterm:  very_low Triangle  {vertexes[24]} {vertexes[25]} {vertexes[26]}\n'+
			f'\tterm:  low      Triangle  {vertexes[27]} {vertexes[28]} {vertexes[29]}\n'+
			f'\tterm:  average  Triangle  {vertexes[30]} {vertexes[31]} {vertexes[32]}\n'+
			f'\tterm:  high     Triangle  {vertexes[33]} {vertexes[34]} {vertexes[35]}\n'+
			'RuleBlock: mamdani\n'+
			'\tdescription: Mamdani Inference for TP\n'+
			'\tenabled: true\n'+
			'\tconjunction: Minimum\n'+
			'\tdisjunction: Maximum\n'+
			'\timplication: Minimum\n'+
			'\tactivation: General\n'+
			f'\trule: if snr is very_poor then tp is high\n'+
			f'\trule: if snr is poor then tp is average\n'+
			f'\trule: if snr is acceptable then tp is low\n'+
			f'\trule: if snr is good then tp is very_low\n'+
			'RuleBlock: mamdani\n'+
			'\tdescription: Mamdani Inference for SF\n'+
			'\tenabled: true\n'+
			'\tconjunction: Minimum\n'+
			'\tdisjunction: Maximum\n'+
			'\timplication: Minimum\n'+
			'\tactivation: General\n'+
			f'\trule: if snr is very_poor then sf is high\n'+
			f'\trule: if snr is poor then sf is average\n'+
			f'\trule: if snr is acceptable then sf is low\n'+
			f'\trule: if snr is good then sf is very_low\n'
		)

def fill_fll3(vertexes, fll='fadr.fll'):	
	filename = \
		f'{ns3_dir}/src/lorawan/examples/{fll}'
	with open(filename, 'w') as file:
		file.write(\
			'Engine: FADR\n' +
			'\tdescription: An Engine for ADR LoRaWAN\n' +
			'InputVariable: snr\n'+
			'\tdescription: Signal Noise Ratio\n'
			'\tenabled: true\n'+
			'\trange: -5.5  27.8\n'+
			'\tlock-range: true\n'+
			f'\tterm: very_poor       Triangle {vertexes[0]} {vertexes[1]} {vertexes[2]}\n'+
			f'\tterm: poor            Triangle {vertexes[3]} {vertexes[4]} {vertexes[5]}\n'+
			f'\tterm: acceptable      Triangle {vertexes[6]} {vertexes[7]} {vertexes[8]}\n'+
			f'\tterm: good            Triangle {vertexes[9]} {vertexes[10]} {vertexes[11]}\n'+
			f'\tterm: very_good       Triangle {vertexes[12]} {vertexes[13]} {vertexes[14]}\n'+
			'OutputVariable: tp\n'+
			'\tdescription: TP based on Mamdani inference\n'+
			'\tenabled: true\n'+
			'\trange:   2  14\n'+
			'\tlock-range: false\n'+
			'\taggregation: Maximum\n'+
			'\tdefuzzifier: Centroid 50\n'+
			'\tdefault: nan\n'+
			'\tlock-previous: false\n'+
			f'\tterm:  very_low  Triangle  {vertexes[15]} {vertexes[16]} {vertexes[17]}\n'+
			f'\tterm:  low       Triangle  {vertexes[18]} {vertexes[19]} {vertexes[20]}\n'+
			f'\tterm:  average   Triangle  {vertexes[21]} {vertexes[22]} {vertexes[23]}\n'+
			f'\tterm:  high      Triangle  {vertexes[24]} {vertexes[25]} {vertexes[26]}\n'+
			f'\tterm:  very_high Triangle  {vertexes[27]} {vertexes[28]} {vertexes[29]}\n'+
			'OutputVariable: sf\n'+
			'\tdescription: SF based on Mamdani inference\n'+
			'\tenabled: true\n'+
			'\trange: 7  12\n'+
			'\tlock-range: false\n'+
			'\taggregation: Maximum\n'+
			'\tdefuzzifier: Centroid 50\n'+
			'\tdefault: nan\n'+
			'\tlock-previous: false\n'+
			f'\tterm:  very_low  Triangle  {vertexes[30]} {vertexes[31]} {vertexes[32]}\n'+
			f'\tterm:  low       Triangle  {vertexes[33]} {vertexes[34]} {vertexes[35]}\n'+
			f'\tterm:  average   Triangle  {vertexes[36]} {vertexes[37]} {vertexes[38]}\n'+
			f'\tterm:  high      Triangle  {vertexes[39]} {vertexes[40]} {vertexes[41]}\n'+
			f'\tterm:  very_high Triangle  {vertexes[42]} {vertexes[43]} {vertexes[44]}\n'+
			'RuleBlock: mamdani\n'+
			'\tdescription: Mamdani Inference for TP\n'+
			'\tenabled: true\n'+
			'\tconjunction: Minimum\n'+
			'\tdisjunction: Maximum\n'+
			'\timplication: Minimum\n'+
			'\tactivation: General\n'+
			f'\trule: if snr is very_poor then tp is very_high\n'+
			f'\trule: if snr is poor then tp is high\n'+
			f'\trule: if snr is acceptable then tp is average\n'+
			f'\trule: if snr is good then tp is low\n'+
			f'\trule: if snr is very_good then tp is very_low\n'+
			'RuleBlock: mamdani\n'+
			'\tdescription: Mamdani Inference for SF\n'+
			'\tenabled: true\n'+
			'\tconjunction: Minimum\n'+
			'\tdisjunction: Maximum\n'+
			'\timplication: Minimum\n'+
			'\tactivation: General\n'+
			f'\trule: if snr is very_poor then sf is very_high\n'+
			f'\trule: if snr is poor then sf is high\n'+
			f'\trule: if snr is acceptable then sf is average\n'+
			f'\trule: if snr is good then sf is low\n'+
			f'\trule: if snr is very_good then sf is very_low\n'
		)

def log(data, log='log-genetic'):
	filename = f'{ns3_dir}/{ag_dir}/results/{log}'
	with open(filename, 'a+') as file:
		file.write(str(data)+'\n\n')

def log_g_bests(data, log='g_bests'):
	filename = f'{ns3_dir}/{ag_dir}/results/{log}'
	with open(filename, 'w') as file:
		s = [str(d) for d in data]
		file.write('\n\n'.join(s))

def log_i_bests(data, log='i_bests'):
	filename = f'{ns3_dir}/{ag_dir}/results/{log}'
	with open(filename, 'w') as file:
		s = [str(d) for d in data]
		file.write('\n\n'.join(s))

def close_file(file):
	file.close()