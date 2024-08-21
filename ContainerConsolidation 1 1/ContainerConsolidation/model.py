import math
import copy
import random
import numpy as np

INFEASIBLE = 100000

class Container():
    def __init__(self, V, limit, verbose=False):
        self.dimensions = V
        self.EMSs = [[np.array((0,0,0)), np.array(V)]]
        self.load_items = []
        self.total_weight = 0  # Track the total weight of boxes in this container
        self.weight_limit = limit

        if verbose:
            print('Init EMSs:', self.EMSs)
    
    def __getitem__(self, index):
        return self.EMSs[index]
    
    def __len__(self):
        return len(self.EMSs)
    
    def update(self, box, selected_EMS, weight, min_vol=1, min_dim=1, verbose=False):
        # 1. Place box in an EMS
        boxToPlace = np.array(box)
        selected_min = np.array(selected_EMS[0])
        ems = [selected_min, selected_min + boxToPlace[:3]]  # Only use dimensions for placement
        self.load_items.append(ems)
        self.total_weight += weight  # Update the total weight

        if verbose:
            print('------------\n*Place Box*:\nEMS:', list(map(tuple, ems)))

        # 2. Generate new EMSs resulting from the intersection of the box
        for EMS in self.EMSs.copy():
            if self.overlapped(ems, EMS):
                
                # eliminate overlapped EMS
                self.eliminate(EMS)
                
                if verbose:
                    print('\n*Elimination*:\nRemove overlapped EMS:',list(map(tuple, EMS)),'\nEMSs left:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
                
                # six new EMSs in 3 dimensionsc
                x1, y1, z1 = EMS[0]; x2, y2, z2 = EMS[1]
                x3, y3, z3 = ems[0]; x4, y4, z4 = ems[1]
                new_EMSs = [
                    [np.array((x4, y1, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y4, z1)), np.array((x2, y2, z2))],
                    [np.array((x1, y1, z4)), np.array((x2, y2, z2))]
                ]
                

                for new_EMS in new_EMSs:
                    new_box = new_EMS[1] - new_EMS[0]
                    isValid = True
                    
                    if verbose:
                        print('\n*New*\nEMS:', list(map(tuple, new_EMS)))

                    # 3. Eliminate new EMSs which are totally inscribed by other EMSs
                    for other_EMS in self.EMSs:
                        if self.inscribed(new_EMS, other_EMS):
                            isValid = False
                            if verbose:
                                print('-> Totally inscribed by:', list(map(tuple, other_EMS)))
                            
                    # 4. Do not add new EMS smaller than the volume of remaining boxes
                    if np.min(new_box) < min_dim:
                        isValid = False
                        if verbose:
                            print('-> Dimension too small.')
                        
                    # 5. Do not add new EMS having smaller dimension of the smallest dimension of remaining boxes
                    if np.product(new_box) < min_vol:
                        isValid = False
                        if verbose:
                            print('-> Volumne too small.')

                    if isValid:
                        self.EMSs.append(new_EMS)
                        if verbose:
                            print('-> Success\nAdd new EMS:', list(map(tuple, new_EMS)))

        if verbose:
            print('\nEnd:')
            print('EMSs:', list(map( lambda x : list(map(tuple,x)), self.EMSs)))
    
    def overlapped(self, ems, EMS):
        if np.all(ems[1] > EMS[0]) and np.all(ems[0] < EMS[1]):
            return True
        return False
    
    def inscribed(self, ems, EMS):
        if np.all(EMS[0] <= ems[0]) and np.all(ems[1] <= EMS[1]):
            return True
        return False
    
    def eliminate(self, ems):
        # numpy array can't compare directly
        ems = list(map(tuple, ems))    
        for index, EMS in enumerate(self.EMSs):
            if ems == list(map(tuple, EMS)):
                self.EMSs.pop(index)
                return
    
    def get_EMSs(self):
        return  list(map( lambda x : list(map(tuple,x)), self.EMSs))
    
    def load(self):
        return np.sum([ np.product(item[1] - item[0]) for item in self.load_items]) / np.product(self.dimensions)
    
    def get_total_weight(self):
        return self.total_weight
    def get_weight_limit(self):
        return self.weight_limit[0]
    # Optionally, print the weight information
    def print_weight_info(self):
        print("Total weight of boxes in container:", self.get_total_weight())
    
class PlacementProcedure():
    def __init__(self, inputs, solution, verbose=False):
        self.Containers = [Container(V, limit) for V, limit in zip(inputs['V'], inputs['limit'])]
        self.boxes = inputs['v']
        self.BPS = np.argsort(solution[:len(self.boxes)])
        self.VBO = solution[len(self.boxes):]
        self.num_opened_containers = 1
        self.inputs = inputs
        self.verbose = verbose
        self.container_details = {}  # Dictionary to store container-wise details

        if self.verbose:
            print('------------------------------------------------------------------')
            print('|   Placement Procedure')
            print('|    -> Boxes:', self.boxes)
            print('|    -> Box Packing Sequence:', self.BPS)
            print('|    -> Vector of Box Orientations:', self.VBO)
            print('-------------------------------------------------------------------')

        self.infisible = False
        self.placement()

    def placement(self):
        items_sorted = [self.boxes[i] for i in self.BPS]

        # Box Selection
        for i, box in enumerate(items_sorted):
            if self.verbose:
                print('Select Box:', box)
                
            # Extract dimensions and weight
            box_dimensions = box[:3]
            box_weight = box[3]

            # Container and EMS selection
            selected_container = None
            selected_EMS = None
            for k in range(self.num_opened_containers):
                # select EMS using DFTRC-2
                EMS = self.DFTRC_2(box_dimensions, k)

                # update selection if "packable"
                if EMS != None:
                    selected_container = k
                    selected_EMS = EMS
                    break

            # Open new empty container
            if selected_container == None:
                self.num_opened_containers += 1
                selected_container = self.num_opened_containers - 1
                if self.num_opened_containers > len(self.Containers):
                    self.infisible = True

                    if self.verbose:
                        print('No more container to open. [Infeasible]')
                    return

                selected_EMS = self.Containers[selected_container].EMSs[0]  # origin of the new container
                if self.verbose:
                    print('No available container... open container', selected_container)

            if self.verbose:
                print('Select EMS:', list(map(tuple, selected_EMS)))

            # Box orientation selection
            BO = self.selected_box_orientaion(self.VBO[i], box_dimensions, selected_EMS)

            # elimination rule for different process
            min_vol, min_dim = self.elimination_rule(items_sorted[i+1:])

            # pack the box to the container & update state information
            self.Containers[selected_container].update(self.orient(box_dimensions, BO), selected_EMS, box_weight, min_vol, min_dim)
            # Store container-wise details
            container_key = f'Container {selected_container + 1}'
            if container_key not in self.container_details:
                self.container_details[container_key] = {'boxes': [], 'orientations': []}

            self.container_details[container_key]['boxes'].append(self.orient(box_dimensions, BO))
            self.container_details[container_key]['orientations'].append(BO)
            if self.verbose:
                print('Add box to Container', selected_container)
                print(' -> EMSs:', self.Containers[selected_container].get_EMSs())
                print('------------------------------------------------------------')
        if self.verbose:
            print('|')
            print('|     Number of used containers:', self.num_opened_containers)
            print('|')
            print('------------------------------------------------------------')
    
    # Distance to the Front-Top-Right Corner
    def DFTRC_2(self, box, k):
        maxDist = -1
        selectedEMS = None

        for EMS in self.Containers[k].EMSs:
            D, W, H = self.Containers[k].dimensions
            for direction in [1,2,3,4,5,6]:
                d, w, h = self.orient(box, direction)
                if self.fitin((d, w, h), EMS):
                    x, y, z = EMS[0]
                    distance = pow(D-x-d, 2) + pow(W-y-w, 2) + pow(H-z-h, 2)

                    if distance > maxDist:
                        maxDist = distance
                        selectedEMS = EMS
        return selectedEMS

    def orient(self, box, BO=1):
        d, w, h = box
        if   BO == 1: return (d, w, h)
        elif BO == 2: return (d, h, w)
        elif BO == 3: return (w, d, h)
        elif BO == 4: return (w, h, d)
        elif BO == 5: return (h, d, w)
        elif BO == 6: return (h, w, d)
    def revert_orient(self, box, BO=1):
        d, w, h = box
        if   BO == 1: return (d, w, h)
        elif BO == 2: return (d, h, w)
        elif BO == 3: return (w, d, h)
        elif BO == 4: return (h, d, w)
        elif BO == 5: return (w, h, d)
        elif BO == 6: return (h, w, d)
    def print_placement(self):
        print('\nPlacement Details:')
        for k in range(self.num_opened_containers):
            print(f'\nContainer {k + 1}:')
            container = self.Containers[k]
            for i, item in enumerate(container.load_items):
                box = item[1] - item[0]
                orientation = self.VBO[i]
                BOs = [1, 2, 3, 4, 5, 6]
                orientation = BOs[math.ceil(orientation*len(BOs))-1]
                if   orientation == 1: ort = "lwh"
                elif orientation == 2: ort = "lhw"
                elif orientation == 3: ort = "wlh"
                elif orientation == 4: ort = "whl"
                elif orientation == 5: ort = "hlw"
                elif orientation == 6: ort = "hwl"
                print(f'Box: {box}', f' positioned at {item[0]} inches from the inner wall.')
        
    def selected_box_orientaion(self, VBO, box, EMS):
        # feasible direction
        BOs = []
        for direction in [1,2,3,4,5,6]:
            if self.fitin(self.orient(box, direction), EMS):
                BOs.append(direction)
        
        # choose direction based on VBO vector
        selectedBO = BOs[math.ceil(VBO*len(BOs))-1]
        
        if self.verbose:
            print('Select VBO:', selectedBO,'  (BOs',BOs, ', vector', VBO,')')
        return selectedBO
         
    def fitin(self, box, EMS):
        # all dimension fit
        for d in range(3):
            if box[d] > EMS[1][d] - EMS[0][d]:
                return False
        return True
    
    def elimination_rule(self, remaining_boxes):
        if len(remaining_boxes) == 0:
            return 0, 0
        
        min_vol = 999999999
        min_dim = 9999
        for box in remaining_boxes:
            # minimum dimension
            dim = np.min(box)
            if dim < min_dim:
                min_dim = dim
                
            # minimum volume
            vol = np.product(box)
            if vol < min_vol:
                min_vol = vol
        return min_vol, min_dim
    
    def evaluate(self):
        if self.infisible:
            return INFEASIBLE
        
        leastLoad = 1
        for k in range(self.num_opened_containers):
            load = self.Containers[k].load()
            weight = self.Containers[k].get_total_weight()
            if weight > self.Containers[k].get_weight_limit():
                return INFEASIBLE
            if load < leastLoad:
                leastLoad = load
        return self.num_opened_containers + leastLoad%1
    def total_vol(self):
        print("Container Space Utilization: ")
        for k in range(self.num_opened_containers):
            load = self.Containers[k].load()
            print("Container: ", k+1, "Volume Utilized: ", round(load*100, 2), "%")
    
    def print_weight_info(self):
        for k in range(self.num_opened_containers):
            print(f"Container {k+1} Total Weight: {self.Containers[k].get_total_weight():.2f} kg")

class BRKGA():
    def __init__(self, inputs, num_generations = 200, num_individuals=120, num_elites = 12, num_mutants = 18, eliteCProb = 0.7, multiProcess = False):
        # Setting
        self.multiProcess = multiProcess
        # Input
        self.inputs =  copy.deepcopy(inputs)
        self.N = len(inputs['v'])
        
        # Configuration
        self.num_generations = num_generations
        self.num_individuals = int(num_individuals)
        self.num_gene = 2*self.N
        
        self.num_elites = int(num_elites)
        self.num_mutants = int(num_mutants)
        self.eliteCProb = eliteCProb
        
        # Result
        self.used_containers = -1
        self.solution = None
        self.best_fitness = -1
        self.history = {
            'mean': [],
            'min': []
        }
        
    def decoder(self, solution):
        placement = PlacementProcedure(self.inputs, solution)
        return placement.evaluate()
    
    def cal_fitness(self, population):
        fitness_list = list()

        for solution in population:
            decoder = PlacementProcedure(self.inputs, solution)
            fitness_list.append(decoder.evaluate())
        return fitness_list

    def partition(self, population, fitness_list):
        sorted_indexs = np.argsort(fitness_list)
        return population[sorted_indexs[:self.num_elites]], population[sorted_indexs[self.num_elites:]], np.array(fitness_list)[sorted_indexs[:self.num_elites]]
    
    def crossover(self, elite, non_elite):
        # chance to choose the gene from elite and non_elite for each gene
        return [elite[gene] if np.random.uniform(low=0.0, high=1.0) < self.eliteCProb else non_elite[gene] for gene in range(self.num_gene)]
    
    def mating(self, elites, non_elites):
        # biased selection of mating parents: 1 elite & 1 non_elite
        num_offspring = self.num_individuals - self.num_elites - self.num_mutants
        return [self.crossover(random.choice(elites), random.choice(non_elites)) for i in range(num_offspring)]
    
    def mutants(self):
        return np.random.uniform(low=0.0, high=1.0, size=(self.num_mutants, self.num_gene))
        
    def fit(self, patient = 4, verbose = False):
        # Initial population & fitness
        population = np.random.uniform(low=0.0, high=1.0, size=(self.num_individuals, self.num_gene))
        fitness_list = self.cal_fitness(population)
        
        if verbose:
            print()
            
        # best    
        best_fitness = np.min(fitness_list)
        best_solution = population[np.argmin(fitness_list)]
        self.history['min'].append(np.min(fitness_list))
        self.history['mean'].append(np.mean(fitness_list))
        
        
        # Repeat generations
        best_iter = 0
        for g in range(self.num_generations):

            # early stopping
            if g - best_iter > patient:
                self.used_containers = math.floor(best_fitness)
                self.best_fitness = best_fitness
                self.solution = best_solution
                if verbose:
                    print()
                return 'feasible'
            
            # Select elite group
            elites, non_elites, elite_fitness_list = self.partition(population, fitness_list)
            
            # Biased Mating & Crossover
            offsprings = self.mating(elites, non_elites)
            
            # Generate mutants
            mutants = self.mutants()

            # New Population & fitness
            offspring = np.concatenate((mutants,offsprings), axis=0)
            offspring_fitness_list = self.cal_fitness(offspring)
            
            population = np.concatenate((elites, mutants, offsprings), axis = 0)
            #fitness_list = elite_fitness_list + offspring_fitness_list
            fitness_list = self.cal_fitness(population)

            # Update Best Fitness
            for fitness in fitness_list:
                if fitness < best_fitness:
                    best_iter = g
                    best_fitness = fitness
                    best_solution = population[np.argmin(fitness_list)]
            
            self.history['min'].append(np.min(fitness_list))
            self.history['mean'].append(np.mean(fitness_list))
            
            if verbose:
                print()
            
        self.used_containers = math.floor(best_fitness)
        self.best_fitness = best_fitness
        self.solution = best_solution
        return 'feasible'