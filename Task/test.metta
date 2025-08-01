!(register-module! ../../GGGP)
!(import! &self GGGP:Task:gggp)



! "TEST: element_at"
!(element_at (10 20 30 40) 0) ;; Expected: 10
!(element_at (10 20 30 40) 2) ;; Expected: 30
!(element_at () 1) ;; Expected: ()


; ! "TEST: is_list and ensure_list"
; !(is_list (expr (const 1))) ;; Expected: True (should match grammar tree)
; !(is_list (const 1)) ;; Expected: False
; !(ensure_list (const 2)) ;; Expected: (const 2)
; !(ensure_list (expr (const 2))) ;; Expected: (expr (const 2))



; ! "TEST: create_individual"
; !(create_individual const 0) ;; Expected: (const 1), (const 2), or (const 3)
; !(create_individual var 0) ;; Expected: (var x)
; !(create_individual op 0) ;; Expected: +, -, *, or /
; !(create_individual expr 0) ;; Expected: full expression, like (expr (const 2) (op +) (var x))



; !(init_population 2)




; ! "TEST: evaluate"
; !(evaluate (expr (const 2)) 4) ;; Expected: 2
; !(evaluate (expr (var x)) 5) ;; Expected: 5
; !(evaluate (expr (expr (const 2)) (op +) ((expr (var x)))) 3) ;; Expected: 5
; !(evaluate (expr (expr (const 6)) (op /) ((expr (const 2)))) 0) ;; Expected: 3
; !(evaluate (expr (expr (const 6)) (op /) ((expr (const 0)))) 0) ;; Expected: 1 (safe divide by zero)


; ! "TEST: fitness"
; !(fitness (expr (const 2))) ;; Expect high error since y = 2x + 1, const ≠ linear
; !(fitness (expr (expr (const 2)) (op +) ((expr (var x))))) ;; Should be low


; ! "TEST: init_population"
; !(init_population 2) ;; Expected: list of 2 expressions
; !(len (init_population 3)) ;; Expected: 3


; ! "TEST: mutate"
; !(mutate (expr (const 1))) ;; Should return a new random expression


; ! "TEST: crossover"
; !(crossover (expr (const 1)) (expr (const 3))) ;; Randomly mixes
; !(crossover (expr (const 2)) (expr (var x))) ;; Expected: mixed expression


; ! "TEST: pick_k"
; !(pick_k (init_population 5) 3) ;; Expected: 3 individuals from population


; ! "TEST: tournament_select"
; !(tournament_select (init_population 5) 2) ;; Expected: one best from 2 randoms



; ! "TEST: evolve_generation"
; !(evolve_generation (init_population 4) 5 2 0.8 0.2 2) ;; 2 new evolved individuals


; ! "TEST: evolve_loop"
; !(evolve_loop (init_population 3) 5 2 0.8 0.2 3) ;; 3 generations


; ;; Final Result

; !(bind! &final_pop (evolve 10 5 3 0.8 0.2 10))
; !(bind! best (best_of &final_pop))
; ! "Best Individual:"
; ! best
; ! "f(-0.5) = " !(evaluate best -0.5)
; ! "f(0.0) = " !(evaluate best 0.0)
; ! "f(0.5) = " !(evaluate best 0.5)
