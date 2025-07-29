;; MeTTa Grammar-Guided Genetic Programming (GGGP)
;; ===============================================

;; GOAL: Symbolic regression for y = 2x + 1

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 1. Grammar Definition
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(bind! &grammar (new-space))

!(add-atom &grammar (expr (expr op expr)))
!(add-atom &grammar (expr var))
!(add-atom &grammar (expr const))

!(add-atom &grammar (op +))
!(add-atom &grammar (op -))
!(add-atom &grammar (op *))
!(add-atom &grammar (op /))

!(add-atom &grammar (var x))

!(add-atom &grammar (const 1))
!(add-atom &grammar (const 2))
!(add-atom &grammar (const 3))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 2. Grammar-Based Individual Creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (create_individual (-> Symbol Number Expression)))
(= (create_individual $symbol $depth)
  (if (>= $depth 5)
    (let (($rules (match &grammar ($symbol $r) $r))
          ($terminals (filter_terminals $rules)))
      (select_random $terminals))
    (let (($rules (match &grammar ($symbol $r) $r)))
      (let (($rule (select_random $rules)))
        (expand_rule $rule (+ $depth 1))))))

(: (filter_terminals (-> List List)))
(= (filter_terminals $rules)
  (filter_list_terminal $rules expr))

(: (filter_list_terminal (-> List Symbol List)))
(= (filter_list_terminal $rules $non_terminal)
  (if (== $rules ()) ()
    (let (($head (car $rules))
          ($tail (cdr $rules)))
      (if (contains_symbol $head $non_terminal)
        (filter_list_terminal $tail $non_terminal)
        (cons $head (filter_list_terminal $tail $non_terminal))))))

(: (contains_symbol (-> Expression Symbol Boolean)))
(= (contains_symbol $tree $sym)
  (if (== $tree $sym)
    True
    (if (is_list $tree)
        (contains_symbol_list $tree $sym)
        False
    )
   )
)

(: (is_list (-> Any Boolean)))
(= (is_list $tree)
  ( let* (
            ($y (match &grammar $tree $tree))
            ($count (size-atom $y))
         ) ( 
          if (== [] $count)
              False
              (if (> $count 1)
                    True
                    False)
            )
    )
)

;; helper function for contains_symbol
(: (contains_symbol_list (-> List Symbol Boolean)))
(= (contains_symbol_list $lst $sym)
  (if (== $lst ()) 
    False
    (if (contains_symbol (car-atom $lst) $sym) 
      True
      (contains_symbol_list (cdr-atom $lst) $sym))
  )
)

(: (expand_rule (-> Expression Number Expression)))
(= (expand_rule $rule $depth)
  (if (is_list $rule)
    (map_expr_create $rule $depth)
    $rule))

(: (map_expr_create (-> List Number List)))
(= (map_expr_create $lst $depth)
  (if (== $lst ()) ()
    (cons (create_individual (car $lst) $depth)
          (map_expr_create (cdr $lst) $depth))))


(: (select_random (-> List Any)))
(= (select_random $lst)
  (element_at $lst (random-int &rng 0 2))
)

(: (element_at (-> List Number Any)))
(= (element_at $lst $i)
  (if (== $i 0) 
    (car-atom $lst)
    (element_at (cdr-atom $lst) (- $i 1))))


; ! "Test: create_individual"
; !(create_individual expr 0)

; ! "Test: filter_terminals"
; !(filter_terminals ((expr const) (expr (expr op expr)) (expr var)))

! "Test: contains_symbol"
!(contains_symbol (expr var) expr) ;;True
!(contains_symbol (const 1) expr)   ;; False

! "Test: element_at 2 index"
!(element_at (10 20 30 40) 2)

! "Random Selection from given list"
!(select_random (10 20 30 40 50))             

;; test case and expected output 
;; True
! "Print True if it Listed in Atomspace, otherway Fasle:"
! (is_list (const 2))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 3. Expression Evaluation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (evaluate (-> Expression Number Number)))
(= (evaluate $tree $x)
  (match $tree
    ((expr $l (op +) $r) (+ (evaluate $l $x) (evaluate $r $x)))
    ((expr $l (op -) $r) (- (evaluate $l $x) (evaluate $r $x)))
    ((expr $l (op *) $r) (* (evaluate $l $x) (evaluate $r $x)))
    ((expr $l (op /) $r)
      (let (($den (evaluate $r $x)))
        (if (== $den 0) 1 (/ (evaluate $l $x) $den))))
    ((var x) $x)
    ((const $v) $v)
    (_ 0))) ;; fallback

; ! "Test: evaluate"
; !(evaluate (expr (const 2) (op +) (var x)) 3)
; !(evaluate (expr (const 6) (op /) (const 2)) 0)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 4. Fitness Function
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (fitness (-> Expression Number)))
(= (fitness $ind)
  (fitness_internal $ind -1 1 5))

(: (fitness_internal (-> Expression Number Number Number Number)))
(= (fitness_internal $ind $x0 $x1 $n)
  (if (== $n 0) 0
    (let (($x (+ $x0 (* (/ (- $x1 $x0) 5) $n)))
          ($y_true (+ (* 2 $x) 1))
          ($y_pred (evaluate $ind $x))
          ($err (* (- $y_pred $y_true) (- $y_pred $y_true))))
      (+ $err (fitness_internal $ind $x0 $x1 (- $n 1))))))

; ! "Test: fitness"
; !(fitness (expr (const 2) (op +) (var x)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 5. Population Initialization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (init_population (-> Number List)))
(= (init_population $n)
  (if (== $n 0) ()
    (cons (create_individual expr 0) (init_population (- $n 1)))))

; ! "Test: init_population"
; !(init_population 3)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 6. Mutation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (mutate (-> Expression Expression)))
(= (mutate $ind)
  (create_individual expr 0))

; ! "Test: mutate"
; !(mutate (expr (const 1) (op +) (const 2)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 7. Crossover
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (crossover (-> Expression Expression Expression)))
(= (crossover $p1 $p2)
  (if (is_list $p1)
    (if (== (random-int &rng 0 2) 0) $p2
      (cons (crossover (car $p1) (car $p2)) (cdr $p1)))
    $p1))

; ! "Test: crossover"
; !(crossover (expr (const 1) (op +) (const 2)) (expr (const 3) (op *) (const 1)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 8. Tournament Selection
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (tournament_select (-> List Number Expression)))
(= (tournament_select $pop $k)
  (let (($choices (pick_k $pop $k)))
    (best_of $choices)))

(: (pick_k (-> List Number List)))
(= (pick_k $pop $k)
  (if (== $k 0) ()
    (cons (element_at $pop (mod (random-int &rng 0 9999) (len $pop)))
          (pick_k $pop (- $k 1)))))

(: (best_of (-> List Expression)))
(= (best_of $lst)
  (if (== (cdr $lst) ()) (car $lst)
    (let (($a (car $lst)))
      (let (($b (best_of (cdr $lst))))
        (if (< (fitness $a) (fitness $b)) $a $b)))))

; ! "Test: tournament_select"
; !(tournament_select (init_population 5) 3)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 9. Evolutionary Loop
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (evolve (-> Number Number Number Number Number Number List)))
(= (evolve $pop_size $depth $tourn_size $cross_p $mut_p $gens)
  (evolve_loop (init_population $pop_size) $depth $tourn_size $cross_p $mut_p $gens))

(: (evolve_loop (-> List Number Number Number Number Number List)))
(= (evolve_loop $pop $depth $tourn_size $cross_p $mut_p $gen)
  (if (== $gen 0) $pop
    (let (($newpop (evolve_generation $pop $depth $tourn_size $cross_p $mut_p (len $pop))))
      (evolve_loop $newpop $depth $tourn_size $cross_p $mut_p (- $gen 1)))))

(: (evolve_generation (-> List Number Number Number Number Number List)))
(= (evolve_generation $pop $depth $tourn_size $cross_p $mut_p $n)
  (if (== $n 0) ()
    (let (
      ($p1 (tournament_select $pop $tourn_size))
      ($p2 (tournament_select $pop $tourn_size))
      ($child (if (< (random-int &rng 0 10) (* $cross_p 10))
                (crossover $p1 $p2) $p1))
      ($mutated (if (< (random-int &rng 0 10) (* $mut_p 10)) (mutate $child) $child))
    )
    (cons $mutated (evolve_generation $pop $depth $tourn_size $cross_p $mut_p (- $n 1))))))

; ! "Test: evolve"
; !(evolve 5 3 2 0.8 0.2 2)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 10. Final Result Output
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; !(bind! final_pop (evolve 20 5 3 0.8 0.2 10))
; !(bind! best (best_of final_pop))
; ! "Best Individual:"
; ! best
; ! "f(-0.5) = " !(evaluate best -0.5)
; ! "f(0.0) = " !(evaluate best 0.0)
; ! "f(0.5) = " !(evaluate best 0.5)
