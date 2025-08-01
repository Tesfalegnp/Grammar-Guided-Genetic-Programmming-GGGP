#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace std;

// Random number generator
random_device rd;
mt19937 gen(rd());

// Node structure for the derivation tree
struct Node {
    string symbol; // Grammar symbol (e.g., "expr", "op")
    string value;  // Terminal value (e.g., "x", "+")
    vector<shared_ptr<Node>> children;

    Node(const string& sym, const string& val = "") : symbol(sym), value(val) {}

    // Print the tree (for debugging)
    void print(int depth = 0) const {
        cout << string(depth * 2, ' ') << symbol;
        if (!value.empty()) cout << ":" << value;
        cout << endl;
        for (const auto& child : children) {
            child->print(depth + 1);
        }
    }
};

// Grammar definition
struct Grammar {
    unordered_map<string, vector<vector<string>>> rules;

    void addRule(const string& lhs, const vector<string>& rhs) {
        rules[lhs].push_back(rhs);
    }

    const vector<vector<string>>& getRules(const string& symbol) const {
        return rules.at(symbol);
    }

    bool isTerminal(const string& symbol) const {
        return rules.find(symbol) == rules.end();
    }
};

// Genetic Programming System
class GGGA {
    Grammar grammar;
    int popSize;
    int maxDepth;
    int tournamentSize;
    double crossoverProb;
    double mutationProb;
    int generations;

    vector<double> X_train;
    vector<double> y_train;
    vector<shared_ptr<Node>> population;

public:
    GGGA(const Grammar& g, int ps, int md, int ts, double cp, double mp, int gen)
        : grammar(g), popSize(ps), maxDepth(md), tournamentSize(ts),
          crossoverProb(cp), mutationProb(mp), generations(gen) {
        // Create training data for f(x) = 2x + 1
        for (double x = -1.0; x <= 1.0; x += 0.1) {
            X_train.push_back(x);
            y_train.push_back(2 * x + 1);
        }
    }

    // Create a random individual using the grow method
    shared_ptr<Node> createIndividual(const string& symbol = "expr", int depth = 0) {
        auto node = make_shared<Node>(symbol);

        // Get all possible rules for this symbol
        const auto& possibleRules = grammar.getRules(symbol);

        // If we've reached max depth, only use terminal rules
        vector<vector<string>> allowedRules;
        if (depth >= maxDepth) {
            for (const auto& rule : possibleRules) {
                bool allTerminals = true;
                for (const auto& s : rule) {
                    if (!grammar.isTerminal(s)) {
                        allTerminals = false;
                        break;
                    }
                }
                if (allTerminals) allowedRules.push_back(rule);
            }
            // If no terminal rules, use any rule
            if (allowedRules.empty()) allowedRules = possibleRules;
        } else {
            allowedRules = possibleRules;
        }

        // Randomly select a rule
        uniform_int_distribution<> ruleDist(0, allowedRules.size() - 1);
        const auto& selectedRule = allowedRules[ruleDist(gen)];

        // Expand each symbol in the rule
        for (const auto& s : selectedRule) {
            if (grammar.isTerminal(s)) {
                node->children.push_back(make_shared<Node>(s, s));
            } else {
                node->children.push_back(createIndividual(s, depth + 1));
            }
        }

        return node;
    }

    // Evaluate the tree for a given x
    double evaluate(const shared_ptr<Node>& node, double x) {
        if (node->symbol == "expr") {
            if (node->children.size() == 3) {
                // Binary operation: left op right
                double left = evaluate(node->children[0], x);
                string op = node->children[1]->children[0]->value;
                double right = evaluate(node->children[2], x);

                if (op == "+") return left + right;
                if (op == "-") return left - right;
                if (op == "*") return left * right;
                if (op == "/") return right != 0 ? left / right : 1.0;
            } else if (node->children.size() == 1) {
                // Single term
                return evaluate(node->children[0], x);
            }
        } else if (node->symbol == "var") {
            return x;
        } else if (node->symbol == "const") {
            return stod(node->children[0]->value);
        } else if (!node->value.empty()) {
            if (node->value == "x") return x;
            try { return stod(node->value); }
            catch (...) { return 0.0; }
        }
        return 0.0;
    }

    // Calculate fitness (mean squared error)
    double fitness(const shared_ptr<Node>& individual) {
        double totalError = 0.0;
        for (size_t i = 0; i < X_train.size(); i++) {
            double prediction = evaluate(individual, X_train[i]);
            double error = prediction - y_train[i];
            totalError += error * error;
        }
        return totalError / X_train.size();
    }

    // Tournament selection
    shared_ptr<Node> tournamentSelection() {
        vector<shared_ptr<Node>> tournament;
        uniform_int_distribution<> dist(0, popSize - 1);

        // Select tournamentSize random individuals
        for (int i = 0; i < tournamentSize; i++) {
            tournament.push_back(population[dist(gen)]);
        }

        // Find the best one in the tournament
        auto best = tournament[0];
        double bestFitness = fitness(best);

        for (size_t i = 1; i < tournament.size(); i++) {
            double currentFitness = fitness(tournament[i]);
            if (currentFitness < bestFitness) {
                best = tournament[i];
                bestFitness = currentFitness;
            }
        }

        return best;
    }

    // Find all non-terminal nodes in a tree
    void getNonTerminals(const shared_ptr<Node>& node, vector<shared_ptr<Node>>& result) {
        if (!grammar.isTerminal(node->symbol)) {
            result.push_back(node);
            for (const auto& child : node->children) {
                getNonTerminals(child, result);
            }
        }
    }

    // Subtree crossover
    pair<shared_ptr<Node>, shared_ptr<Node>> crossover(shared_ptr<Node> parent1, shared_ptr<Node> parent2) {
        auto child1 = make_shared<Node>(*parent1);
        auto child2 = make_shared<Node>(*parent2);

        // Find all non-terminal nodes in both trees
        vector<shared_ptr<Node>> nodes1, nodes2;
        getNonTerminals(child1, nodes1);
        getNonTerminals(child2, nodes2);

        if (nodes1.empty() || nodes2.empty()) {
            return {child1, child2};
        }

        // Select random crossover points
        uniform_int_distribution<> dist1(0, nodes1.size() - 1);
        uniform_int_distribution<> dist2(0, nodes2.size() - 1);
        auto node1 = nodes1[dist1(gen)];
        auto node2 = nodes2[dist2(gen)];

        // Swap subtrees
        swap(node1->children, node2->children);

        return {child1, child2};
    }

    // Subtree mutation
    shared_ptr<Node> mutate(shared_ptr<Node> individual) {
        // Find all non-terminal nodes
        vector<shared_ptr<Node>> nodes;
        getNonTerminals(individual, nodes);

        if (nodes.empty()) return individual;

        // Select random mutation point
        uniform_int_distribution<> dist(0, nodes.size() - 1);
        auto node = nodes[dist(gen)];

        // Generate new subtree
        node->children.clear();
        auto newSubtree = createIndividual(node->symbol);
        node->children = newSubtree->children;

        return individual;
    }

    // Convert tree to string expression
    string treeToString(const shared_ptr<Node>& node) {
        if (node->symbol == "expr") {
            if (node->children.size() == 3) {
                string left = treeToString(node->children[0]);
                string op = treeToString(node->children[1]);
                string right = treeToString(node->children[2]);
                return "(" + left + " " + op + " " + right + ")";
            } else if (node->children.size() == 1) {
                return treeToString(node->children[0]);
            }
        } else if (node->symbol == "op") {
            return node->children[0]->value;
        } else if (node->symbol == "var") {
            return node->children[0]->value;
        } else if (node->symbol == "const") {
            return node->children[0]->value;
        }
        return node->value;
    }

    // Initialize population
    void initPopulation() {
        population.clear();
        for (int i = 0; i < popSize; i++) {
            population.push_back(createIndividual());
        }
    }

    // Run the evolutionary process
    shared_ptr<Node> evolve() {
        initPopulation();

        shared_ptr<Node> bestIndividual;
        double bestFitness = numeric_limits<double>::max();

        for (int gen = 0; gen < generations; gen++) {
            // Evaluate fitness and find best
            for (const auto& ind : population) {
                double currentFitness = fitness(ind);
                if (currentFitness < bestFitness) {
                    bestFitness = currentFitness;
                    bestIndividual = ind;
                }
            }

            cout << "Generation " << gen + 1 << ": Best Fitness = " << bestFitness << endl;
            cout << "Best Expression: " << treeToString(bestIndividual) << endl;

            // Create new generation
            vector<shared_ptr<Node>> newPopulation;
            uniform_real_distribution<> probDist(0.0, 1.0);

            while (newPopulation.size() < popSize) {
                auto parent1 = tournamentSelection();
                auto parent2 = tournamentSelection();

                shared_ptr<Node> child1, child2;

                if (probDist(gen) < crossoverProb) {
                    auto children = crossover(parent1, parent2);
                    child1 = children.first;
                    child2 = children.second;
                } else {
                    child1 = parent1;
                    child2 = parent2;
                }

                if (probDist(gen) < mutationProb) {
                    child1 = mutate(child1);
                }
                if (probDist(gen) < mutationProb) {
                    child2 = mutate(child2);
                }

                newPopulation.push_back(child1);
                if (newPopulation.size() < popSize) {
                    newPopulation.push_back(child2);
                }
            }

            population = move(newPopulation);
        }

        return bestIndividual;
    }
};

int main() {
    // Define the grammar
    Grammar grammar;
    grammar.addRule("expr", {"expr", "op", "expr"});
    grammar.addRule("expr", {"var"});
    grammar.addRule("expr", {"const"});
    grammar.addRule("op", {"+"});
    grammar.addRule("op", {"-"});
    grammar.addRule("op", {"*"});
    grammar.addRule("op", {"/"});
    grammar.addRule("var", {"x"});
    grammar.addRule("const", {"1"});
    grammar.addRule("const", {"2"});
    grammar.addRule("const", {"3"});
    grammar.addRule("const", {"4"});
    grammar.addRule("const", {"5"});

    // Create and run the GGGA system
    GGGA ggga(grammar, 50, 5, 3, 0.8, 0.2, 20);
    auto bestSolution = ggga.evolve();

    // Show results
    cout << "\n=== Evolutionary Results ===" << endl;
    cout << "Final Expression: " << ggga.treeToString(bestSolution) << endl;
    cout << "Target Function: f(x) = 2x + 1" << endl;

    // Test the solution
    cout << "\nTest Predictions:" << endl;
    vector<double> testValues = {-0.5, 0.0, 0.5};
    for (double x : testValues) {
        double prediction = ggga.evaluate(bestSolution, x);
        cout << "f(" << x << ") = " << prediction << " (Expected: " << 2*x + 1 << ")" << endl;
    }

    return 0;
}