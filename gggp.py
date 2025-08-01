import random
import numpy as np
import copy

class Node:
    """Represents a node in the derivation tree"""
    def __init__(self, symbol, value=None):
        self.symbol = symbol    # Grammar symbol (e.g., 'expr', 'op')
        self.value = value      # Terminal value (e.g., 'x', '+')
        self.children = []      # Child nodes

    def __repr__(self, level=0):
        """String representation of the tree"""
        ret = "  " * level + f"{self.symbol}:{self.value if self.value else ''}\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class GGGP:
    """Grammar-Guided Genetic Programming System"""
    def __init__(self, grammar, pop_size=50, max_depth=5, tournament_size=3, 
                 crossover_prob=0.8, mutation_prob=0.2, generations=20):
        self.grammar = grammar
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.generations = generations
        self.population = []
        self.fitness_history = []
         
        # Training data: f(x) = 2x + 1
        self.X_train = np.linspace(-1, 1, 20)
        self.y_train = 2 * self.X_train + 1

    def create_individual(self, symbol='expr', depth=0):
        """Create a new individual using grow method"""
        node = Node(symbol)
        
        # Base case: max depth reached
        if depth >= self.max_depth:
            # Select only terminal rules
            terminal_rules = [rule for rule in self.grammar[symbol] 
                             if all(s not in self.grammar for s in rule)]
            if not terminal_rules:
                terminal_rules = self.grammar[symbol]
            rule = random.choice(terminal_rules)
        else:
            rule = random.choice(self.grammar[symbol])
        
        # Expand the rule
        for s in rule:
            if s in self.grammar:
                # Non-terminal: recursive expansion
                child = self.create_individual(s, depth + 1)
                node.children.append(child)
            else:
                # Terminal: create leaf node
                node.children.append(Node(s, s))
            # print (f"experision are :  {node}")    
        return node

    def evaluate(self, node, x):
        """Evaluate the tree for a given x value"""
        try:
            if node.symbol == 'expr' and len(node.children) == 3:
                # Binary operation: left op right
                left = self.evaluate(node.children[0], x)
                op = node.children[1].children[0].value if node.children[1].children else node.children[1].value
                right = self.evaluate(node.children[2], x)
                
                if left is None or right is None:
                    return None
                
                if op == '+': return left + right
                if op == '-': return left - right
                if op == '*': return left * right
                if op == '/': 
                    return left / right if right != 0 else None
                
            elif node.symbol == 'expr' and len(node.children) == 1:
                # Single term (var or const)
                return self.evaluate(node.children[0], x)
                
            elif node.symbol == 'op':
                # Operator node
                return node.children[0].value if node.children else node.value
                
            elif node.symbol == 'var':
                # Variable node
                return x
                
            elif node.symbol == 'const':
                # Constant node
                try:
                    return float(node.children[0].value) if node.children else float(node.value)
                except:
                    return None
                
            else:  # Terminal node
                if node.value == 'x': return x
                try: return float(node.value)
                except: return None
                
        except Exception:
            return None

    def fitness(self, individual):
        """Calculate mean squared error (MSE) fitness"""
        predictions = []
        for x in self.X_train:
            pred = self.evaluate(individual, x)
            if pred is None:
                return float('inf')  # Penalize invalid expressions
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mse = np.mean((predictions - self.y_train) ** 2)
        return mse

    def tournament_selection(self):
        """Select individual using tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        tournament.sort(key=lambda ind: self.fitness(ind))
        return copy.deepcopy(tournament[0])

    def crossover(self, parent1, parent2):
        """Subtree crossover between two parents"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Find crossover points (non-terminal nodes)
        nodes1 = self.get_non_terminals(child1)
        nodes2 = self.get_non_terminals(child2)
        
        if not nodes1 or not nodes2:
            return child1, child2
        
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)
        
        # Swap subtrees
        node1.children, node2.children = node2.children, node1.children
        
        return child1, child2

    def mutate(self, individual):
        """Subtree mutation"""
        # Find mutation points (non-terminal nodes)
        nodes = self.get_non_terminals(individual)
        if not nodes:
            return individual
        
        node = random.choice(nodes)
        
        # Generate new subtree
        new_subtree = self.create_individual(node.symbol, depth=0)
        node.children = new_subtree.children
        
        return individual

    def get_non_terminals(self, node):
        """Get all non-terminal nodes in tree"""
        nodes = []
        if node.symbol in self.grammar:
            nodes.append(node)
            for child in node.children:
                nodes.extend(self.get_non_terminals(child))
        return nodes

    def init_population(self):
        """Initialize population"""
        self.population = [self.create_individual() for _ in range(self.pop_size)]

    def evolve(self):
        """Run evolutionary process"""
        self.init_population()
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitnesses = []
            for ind in self.population:
                fitness = self.fitness(ind)
                fitnesses.append(fitness)
            
            best_fitness = min(fitnesses)
            self.fitness_history.append(best_fitness)
            
            # Find best individual
            best_idx = np.argmin(fitnesses)
            best_individual = self.population[best_idx]
            
            print(f"Gen {gen+1}: Best Fitness = {best_fitness:.4f}")
            print(f"Best Expression: {self.tree_to_string(best_individual)}")
            
            # Create new generation
            new_population = []
            
            while len(new_population) < self.pop_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_prob:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                if random.random() < self.mutation_prob:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_prob:
                    child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.pop_size]
        
        return best_individual

    def tree_to_string(self, node):
        """Convert tree to mathematical expression string"""
        if node.symbol == 'expr' and len(node.children) == 3:
            left = self.tree_to_string(node.children[0])
            op = self.tree_to_string(node.children[1])
            right = self.tree_to_string(node.children[2])
            return f"({left} {op} {right})"
            
        elif node.symbol == 'expr' and len(node.children) == 1:
            return self.tree_to_string(node.children[0])
            
        elif node.symbol == 'op':
            return node.children[0].value if node.children else node.value
            
        elif node.symbol == 'var':
            return node.children[0].value if node.children else node.value
            
        elif node.symbol == 'const':
            return node.children[0].value if node.children else node.value
            
        else:  # Terminal node
            return str(node.value)

# Define grammar for arithmetic expressions
grammar = {
    'expr': [
        ['expr', 'op', 'expr'],  # Binary operation
        ['var'],                 # Variable
        ['const']                # Constant
    ],
    'op': [
        ['+'], 
        ['-'], 
        ['*'], 
        ['/']
    ],
    'var': [
        ['x']  # Only variable is 'x'
    ],
    'const': [
        [str(i)] for i in range(1, 6)  # Constants 1-5
    ]
}

# Run GGGP
gggp = GGGP(
    grammar,
    pop_size=50,
    max_depth=5,
    tournament_size=3,
    crossover_prob=0.8,
    mutation_prob=0.2,
    generations=20
)

best_solution = gggp.evolve()

# Final results
print("\n=== Evolutionary Results ===")
print(f"Final Expression: {gggp.tree_to_string(best_solution)}")
print(f"Target Function: f(x) = 2x + 1")

# Evaluate on test data
test_x = np.array([-0.5, 0, 0.5])
print("\nTest Predictions:")
for x in test_x:
    pred = gggp.evaluate(best_solution, x)
    expected = 2*x + 1
    print(f"f({x:.1f}) = {pred if pred is not None else 'Invalid'}, Expected: {expected:.1f}")