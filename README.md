
# 🌱 Grammar-Guided Genetic Programming in MeTTa (GGGP)

This project implements a full **Grammar-Guided Genetic Programming (GGGP)** system using the [MeTTa language](https://github.com/opencog/meTTa), with the goal of discovering symbolic expressions that approximate the target function:

> **Target Function**:  `y = 2x + 1`

It uses a context-free grammar to generate expression trees, applies genetic operations (mutation, crossover), and evaluates individuals based on how well they approximate the target function over a sampled domain.

---

## 🧩 Features

- ✅ Symbolic regression via evolutionary search
- ✅ Custom grammar for valid mathematical expression trees
- ✅ Recursively defined tree-based individuals
- ✅ Expression evaluation engine with safe division
- ✅ Fitness calculation based on squared error over multiple test points
- ✅ Population initialization, mutation, crossover
- ✅ Tournament selection
- ✅ Full evolutionary loop

---

## 📁 Project Structure

```

gggp/
│
├── GGGP.metta       # Full MeTTa code
├── README.md        # This documentation

````

---

## 📖 How It Works

### 1. 🎓 Grammar Definition

Defines valid production rules for expression construction:

```metta
(expr (expr op expr))
(expr var)
(expr const)

(op +) (op -) (op *) (op /)
(var x)
(const 1) (const 2) (const 3)
````

---

### 2. 🧱 Individual Creation

The function `create_individual` recursively creates valid expressions using the grammar, with depth control.

```metta
!(create_individual expr 0)
```

Output: Something like `(expr (const 2 ()) (op +) ((expr (var x ()))))`

---

### 3. 🧮 Evaluation

The `evaluate` function computes the result of an expression given an input value for `x`.

```metta
!(evaluate (expr (const 2 ()) (op +) ((expr (var x ())))) 3)
;; Result: 5
```

---

### 4. 📏 Fitness Function

Measures squared error over a set of points in the interval `[-1, 1]`:

```metta
!(fitness (expr (const 2 ()) (op +) ((expr (var x ())))))
```

---

### 5. 🧬 Genetic Operators

* **Mutation**: Creates a new individual from scratch.
* **Crossover**: Combines subtrees from two parents.
* **Tournament Selection**: Picks best from `k` randomly sampled individuals.

---

### 6. 🔁 Evolution Loop

```metta
!(evolve 10 5 3 0.8 0.2 20)
```

Parameters:

* `10` individuals
* `5` max depth
* `3` tournament size
* `0.8` crossover probability
* `0.2` mutation probability
* `20` generations

---

## 🧪 Sample Test Commands

```metta
!(create_individual const 0)
!(create_individual var 0)
!(create_individual op 0)
!(create_individual expr 0)

!(evaluate (expr (const 2 ()) (op +) ((expr (var x ())))) 1) ; ➜ 3
!(fitness (expr (const 2 ()) (op +) ((expr (var x ())))))
!(init_population 5)

!(mutate (expr (const 1 ()) (op +) ((expr (const 2 ())))))
!(crossover (expr (const 1 ()) (op +) ((expr (const 2 ())))) (expr (var x ())))

!(tournament_select (init_population 5) 2)
!(evolve 10 5 3 0.8 0.2 5)
```

---

## 🚀 Running Full Evolution

```metta
!(bind! &final_pop (evolve 20 5 3 0.8 0.2 10))
!(bind! best (best_of &final_pop))

! "Best Individual:"
! best

! "Test Results:"
!( "f(-0.5) = " (evaluate best -0.5))
!( "f(0.0) = " (evaluate best 0.0))
!( "f(0.5) = " (evaluate best 0.5))
```

---

## 📋 Dependencies

* [MeTTa Language](https://github.com/opencog/meTTa)
* Random number generator (ensure `&rng` is bound to a `(random-generator)`)

```metta
!(bind! &rng (random-generator))
```

---

## 📚 References

* [Genetic Programming](https://en.wikipedia.org/wiki/Genetic_programming)
* [Symbolic Regression](https://en.wikipedia.org/wiki/Symbolic_regression)
* [OpenCog MeTTa](https://wiki.opencog.org/MeTTa)

---

## ✍️ Author

Hope @ iCog Labs
Trained & supported by OpenAI ChatGPT

---

## 🧠 Example Output (Expected)

A near-optimal individual after evolution:

```
(expr (const 2 ()) (op *) ((expr (var x ()))))
```

Evaluated:

```
x = -0.5 → 0
x =  0.0 → 1
x =  0.5 → 2
```
