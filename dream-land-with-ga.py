import sys
import random
from statistics import mean
from collections import deque
from enum import IntEnum
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class IllegalSolutionError(Exception):
    pass


class DreamLand():
    class Atraction(IntEnum):
        entrance = 0
        cafe = 1
        boat = 2
        cups = 3
        restaurant = 4
        ferris_wheel = 5
        haunted_house = 6
        roller_coaster = 7
        maze = 8
    
    _TRAVEL_TIME = [
        [-1, 9,-1, 7,12,-1,-1,-1,-1],
        [ 9,-1,11,12, 7,13,-1,-1,-1],
        [-1,11,-1,-1,14, 8,-1,-1,-1],
        [ 7,12,-1,-1,11,-1, 7,12,-1],
        [12, 7,14,11,-1, 9,13, 9,13],
        [-1,13, 8,-1, 9,-1,-1,13, 7],
        [-1,-1,-1, 7,13,-1,-1, 7,-1],
        [-1,-1,-1,12, 9,13, 7,-1, 6],
        [-1,-1,-1,-1,13, 7,-1, 6,-1]]
    
    _TIME_LIMIT = 200
    
    def __init__(self):
        self.satisfaction_level = [0,50,36,45,79,55,63,71,42]
        self.duration = [0,20,28,15,35,17,18,14,22]
        self.position = self.Atraction.entrance
        self.satisfaction = 0
        self.consumption_time = 0
        self.route = [self.Atraction.entrance]
    
    def _move(self, move):
        try:
            self.Atraction(move)
        except ValueError:
            raise IllegalSolutionError('No existing atraction.')
        if self._TRAVEL_TIME[self.position][move] == -1:
            raise IllegalSolutionError('No existing path.')
        else:
            self.consumption_time += self._TRAVEL_TIME[self.position][move]
            self.consumption_time += self.duration[move]
            self.position = move
            self.route.append(self.Atraction(move))
        if self.consumption_time > self._TIME_LIMIT:
            raise IllegalSolutionError('The time is up.')
        if self.satisfaction_level[move] >= 0:
            self.satisfaction += self.satisfaction_level[move]
            self.duration[move] = 0
            self.satisfaction_level[move] = -1
        else:
            raise IllegalSolutionError('Have been visited.')
        return self.position == self.Atraction.entrance
    
    def _moves(self, moves):
        try:
            for move in moves:
                if self._move(move):
                    break
            else:
                raise IllegalSolutionError('You are lost.')
        except IllegalSolutionError:
            raise
        return self.satisfaction, (self.consumption_time, tuple(self.route))
    
    def solve(self, solution):
        try:
            score, info = self._moves(solution)
        except IllegalSolutionError:
            raise
        return score, info


class Individual():
    """Individual that stands for the answer of the problem"""
    _GENOME_SIZE = 36
    _GENE_SIZE = 4
    _GENE_MASK = (2 << (_GENE_SIZE -1)) - 1
    _MUTATE_PROBABILITY = 0.05
    
    def __init__(self, genome=None):
        if genome is None:
            self._genome = random.getrandbits(self._GENOME_SIZE)
        else:
            assert 0 <= genome < 2 ** self._GENOME_SIZE
            self._genome = genome
        
        self._problem = DreamLand()
        self._solution = list(islice(self._genome_slicer(),
                                     self._GENOME_SIZE // self._GENE_SIZE))
        try:
            self._fitness, self._info = self._problem.solve(self._solution)
        except IllegalSolutionError as e:
            self._fitness = -1
            self._info = (str(e),)
        
    def get_genome(self):
        return self._genome
    
    def get_fitness(self):
        return self._fitness
        
    def _genome_slicer(self):
        genome = self._genome
        while True:
            gene = genome & self._GENE_MASK
            genome = genome >> self._GENE_SIZE
            yield gene
    
    def _is_valid_operand(self, other):
        return hasattr(other, 'get_fitness')
    
    def __hash__(self):
        return hash((type(self), self._fitness, self._info))
    
    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self._fitness == other.get_fitness()
    
    def __ne__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self._fitness != other.get_fitness()
    
    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self._fitness < other.get_fitness()
    
    def __le__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self._fitness <= other.get_fitness()
    
    def __gt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self._fitness > other.get_fitness()
    
    def __ge__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self._fitness >= other.get_fitness()
    
    def __str__(self):
        return f'{self._fitness}: {self._info}'
    
    def mate(self, other):
        """Give birth to a child"""
        assert isinstance(other, Individual)
        
        child_genome = 0
        self_genome = self._genome
        other_genome = other.get_genome()
        
        # inherits from parents
        mask_mate = random.getrandbits(self._GENOME_SIZE)
        self_genome = self_genome & mask_mate
        other_genome = other_genome & ~mask_mate
        child_genome = self_genome | other_genome
        
        # genetic mutation
        mask_mutation = 0
        for _ in range(self._GENOME_SIZE):
            mask_mutation = mask_mutation << 1
            if random.random() <= self._MUTATE_PROBABILITY:
                mask_mutation = mask_mutation | 0b1
        child_genome = child_genome ^ mask_mutation
        
        return Individual(child_genome)
        
        
class Population():
    """Group of individuals"""
    _FERTILITY_RATE = 10
    _CATASTOROPHE_DAMAGE = 100
    
    def __init__(self, population_size):
        self._population_size = population_size
        self.generation = [Individual() for _ in range(self._population_size)]
        self.generation.sort(reverse=True)
        self.generation_number = 0
    
    def next_generation(self):
        self.generation_number += 1
        
        # divide individuals into elites and non-elites
        pareto = self._population_size // 5
        elites = self.generation[: pareto]
        non_elites = self.generation[pareto :]
        
        # all the elite to have a chance to marry non-elite
        children = []
        for parent1 in elites:
            parent2 = random.choice(non_elites)
            for _ in range(self._FERTILITY_RATE):
                children.append(parent1.mate(parent2))
        
        # choose individuals to survive
        elites = random.sample(elites, 12 * len(elites) // 15)
        non_elites = random.sample(non_elites, 3 * len(non_elites) // 15)
        self.generation = elites + children + non_elites
        self.generation.sort(reverse=True)
        self.generation = self.generation[: self._population_size]
        
        # logging the generation
        min_fitness = self.generation[-1].get_fitness()
        max_fitness = self.generation[0].get_fitness()
        mean_fitness = mean(i.get_fitness() for i in self.generation)
        median_fitness = self.generation[self._population_size // 2].get_fitness()
        
        return Aspect(min_fitness, max_fitness, mean_fitness, median_fitness)
    
    def catastrophe(self):
        """Some kind of natural disaster that would cause a wider evolution"""
        survivor_num = self._population_size // self._CATASTOROPHE_DAMAGE
        survivor = random.sample(self.generation, survivor_num)
        newcomer = [Individual() for _ in range(self._population_size)]
        self.generation = survivor + newcomer
        self.generation.sort(reverse=True)
        self.generation = self.generation[: self._population_size]


class Aspect():
    """Aspect of the evolution"""
    def __init__(self, min, max, mean, median):
        self.min = min
        self.max = max
        self.mean = mean
        self.median = median


class Data4Graph():
    """Store of the data for drawing a graph"""
    def __init__(self):
        self.min = []
        self.max = []
        self.mean = []
        self.median = []
    
    def append(self, aspect):
        assert isinstance(aspect, Aspect)
        self.min.append(aspect.min)
        self.max.append(aspect.max)
        self.mean.append(aspect.mean)
        self.median.append(aspect.median)
    
    def check(self):
        return (len(self.min)
                == len(self.max) == len(self.mean) == len(self.median))


def visualize(data4graph):
    """Draw a graph"""
    assert isinstance(data4graph, Data4Graph)
    assert data4graph.check()
    x = range(1, len(data4graph.min) + 1)
    plt.figure()
    plt.plot(x, data4graph.min, marker='.', label='min_fitness')
    plt.plot(x, data4graph.max, marker='.', label='max_fitness')
    plt.plot(x, data4graph.mean, marker='.', label='mean_fitness')
    plt.plot(x, data4graph.median, marker='.', label='median_fitness')
    plt.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=10)
    plt.grid()
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('Dream land problem')
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.show()


def is_ipython():
    try:
        get_ipython()
    except:
        result = False
    else:
        result = True
    return result


class OutputManager():
    """Managing output to the stdin"""
    def __init__(self, verbose):
        self._verbose = verbose
        self._ipython = is_ipython()
    
    _arrow = {0: '\r↑', 1: '\r→', 2: '\r↓', 3: '\r←'}
    
    def _clear_output(self):
        if self._ipython:
            # carriage return only
            s = '\r'
        else:
            # erase in line and carriage return
            s = '\033[2K\033[G'
        sys.stdout.write(s)
        sys.stdout.flush()
    
    def continue_(self, generation_number, aspect_max):
        if self._verbose == 0:
            if not self._ipython: self._clear_output()
            if self._ipython:
                s = self._arrow[generation_number % 4]
            else:
                s = f'{generation_number}: {aspect_max}'
            sys.stdout.write(s)
            sys.stdout.flush()
        elif self._verbose > 0:
            print(f'{generation_number}: {aspect_max}')
    
    def catastrophe(self):
        if self._verbose == 0:
            if self._ipython:
                s = '\r※'
            else:
                self._clear_output()
                s = 'CATASTROPHE OCCURED!'
            sys.stdout.write(s)
            sys.stdout.flush()
        elif self._verbose > 0:
            print('CATASTROPHE OCCURED!')
    
    def epoch_over(self, solutions):
        if self._verbose == 0:
            self._clear_output()
        for i, solution in enumerate(solutions[0:10]):
            print(f'No.{i+1} score is {solution} with genome {solution.get_genome()}')


class EvolutionController():
    """Some existence that controlls the evolution"""
    def __init__(self, population_size=100, epochs=10000,
                 patience=20, verbose=0, graph=False):
        self._population = Population(population_size)
        self._epochs = epochs
        self._patience = patience 
        self._memory = deque([], patience)
        self._graph = graph
        if self._graph:
            self._data4graph = Data4Graph()
        self._outmgr = OutputManager(verbose)
        self._solutions = []
    
    def start(self):
        """Start the evolution"""
        for epoch in range(1, self._epochs + 1):
            aspect = self._population.next_generation()
            self._solutions.append(self._population.generation[0])
            self._outmgr.continue_(self._population.generation_number,
                                   aspect.max)
            if self._graph:
                self._data4graph.append(aspect)
            
            # catastrophe check
            self._memory.append(aspect.max)
            if self._memory.count(self._memory[-1]) == self._patience:
                self._outmgr.catastrophe()
                self._population.catastrophe()
        else:
            solution_unique = set(self._solutions)
            solutions = sorted(list(solution_unique), reverse=True)
            self._outmgr.epoch_over(solutions)
            if self._graph:
                visualize(self._data4graph)


def main():
    ec = EvolutionController(verbose=0, graph=True)
    ec.start()


if __name__ == '__main__':
    main()
