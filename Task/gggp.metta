;; Full Grammar-Guided Genetic Programming (GGGP) in MeTTa
;; Target: Symbolic Regression for y = 2x + 1

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

!(bind! &max_depth 5)
; !(bind! &rng (random-generator))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 2. Helper Functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(=(len $pop) (size-atom $pop))

(: (element_at (-> List Number Any)))

(= (element_at () $i) ())
(= (element_at $lst $i)
  (if (== $i 0)
      (car-atom $lst)
      (element_at (cdr-atom $lst) (- $i 1))))

(: (select_random (-> List Any)))
(= (select_random $lst)
  (element_at $lst (random-int &rng 0 (size-atom $lst))))

(: (is_list (-> Any Boolean)))
(= (is_list $tree)
   (let* (
           ($y (collapse (let $x (get-atoms &grammar) $x)))
           ($bool (is_list_helper $y $tree)))
     $bool))

(= (is_list_helper () $search_tree) False)
(= (is_list_helper $All_List $search_tree)
  (let* (($first (car-atom $All_List))
         ($remain (cdr-atom $All_List)))
    (if (== $first $search_tree)
      True
      (is_list_helper $remain $search_tree))))

      ;; to be ensure the the atom in atomspace
(: (ensure_list (-> Any List)))
(= (ensure_list $r)
  (if (is_list $r) 
      $r 
      (cons-atom $r ())
  ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 3. Individual Creation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (create_individual (-> Symbol Number Expression)))

(= (create_individual expr $depth)
  (let* (
         ($rules (collapse (match &grammar (expr $r) $r)))
         ($terminals (var (const (op))))
         ($rule (if (>= $depth &max_depth)
                    (select_random $terminals)
                    (select_random $rules)))
        )
        ( 
          if (== $rule (expr op expr))
            (let*  ( ($l (create_individual expr (+ $depth 1)))
                    ($o (create_individual op (+ $depth 1)))
                    ($r (create_individual expr (+ $depth 1))) )
                  (cons-atom expr ($l ( $o ($r))))
                )
            (if (== $rule var)
                (let* (($v (create_individual var (+ $depth 1))))
                    (cons-atom expr ($v)) )
                (let* (($c (create_individual const (+ $depth 1))))
                    (cons-atom expr ($c)) ))
        )
  )
)

(= (create_individual var $depth)
  (let* (
         ($rhs (match &grammar (var $v) $v))
         ($v (select_random ($rhs)))
        )
        ( cons-atom var ($v ()) )))

(= (create_individual op $depth)
  ( let* (
          ($all (collapse (match &grammar (op $x) (op $x))))
          ($c (select_random $all))
        ) $c ))

(= (create_individual const $depth)
  ( let* (
          ($all (collapse (match &grammar (const $x) $x)))
          ($c (select_random $all))
        ) (cons-atom const ($c ())) ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 4. Evaluation and Fitness
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (evaluate (-> Expression Number Number)))
(= (evaluate $tree $x)
  (match $tree
    ((expr $l (op +) $r) (+ (evaluate $l $x) (evaluate $r $x)))
    ((expr $l (op -) $r) (- (evaluate $l $x) (evaluate $r $x)))
    ((expr $l (op *) $r) (* (evaluate $l $x) (evaluate $r $x)))
    ((expr $l (op /) $r)
      (let* (($den (evaluate $r $x)))
           (if (== $den 0) 
                1   
                (/ (evaluate $l $x) $den))
      ))
    ((expr (var x)) $x)
    ((expr (const $v)) $v)
    ((var x) $x)
    ((const $v) $v)
    (_ 0)
  )
)

(: (fitness (-> Expression Number)))
(= (fitness $ind)
  (fitness_internal $ind -1 1 5))

(: (fitness_internal (-> Expression Number Number Number Number)))
(= (fitness_internal $ind $x0 $x1 $n)
  (if (== $n 0) 
         0
        (let* (
            ($x (+ $x0 (* (/ (- $x1 $x0) 5) $n)))  ;; X= Xo+ ( (X2-X1)/5 )n general forumal seems like this
            ($y_true (+ (* 2 $x) 1))    ;; y_true= 2x+1 
            ($y_pred (evaluate $ind $x))
            ($err (* (- $y_pred $y_true) (- $y_pred $y_true)))  ;;err=  (x1-x2)^2
            )
            (+ $err (fitness_internal $ind $x0 $x1 (- $n 1)))
        )
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 5. Population and Genetic Operators
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (init_population (-> Number List)))
(= (init_population $n)
  (if (== $n 0) 
    ()
    (let* 
        (($created (create_individual expr 0))
          ($new_n (- $n 1))
          ($new_init (init_population $new_n))
          )
        (cons-atom $created $new_init))
  )
)

(: (mutate (-> Expression Expression)))
(= (mutate $ind) (create_individual expr 0))

(: (crossover (-> Expression Expression Expression)))
(= (crossover $p1 $p2)
  (if (is_list $p1)
      (if (== (random-int &rng 0 2) 0) 
          $p2
          (cons-atom (crossover (car-atom $p1) (car-atom $p2)) (cdr-atom $p1))
        )
        $p1
    )
)

(: (tournament_select (-> List Number Expression)))
(= (tournament_select $pop $k)
  (let* (($choices (pick_k $pop $k))) 
        (best_of $choices) ;;best_of
  )
)


        ;; pick the element from the give list randomly with element_at function
(: (pick_k (-> List Number List)))
(= (pick_k $pop $k)
  (if (== $k 0) 
      ()
      (cons-atom (element_at $pop (random-int &rng 0 (size-atom $pop)))
          (pick_k $pop (- $k 1)))))

(: (best_of (-> List Expression)))

(= (best_of ()) ())
(= (best_of $lst)
  (if (== (cdr-atom $lst) ()) 
        (car-atom $lst)
        (let*
            (($a (car-atom $lst)) 
             ($b (best_of (cdr-atom $lst)))
            )
            (if (< (fitness $a) (fitness $b)) 
                        $a 
                        $b
            )
        )
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 6. Evolutionary Loop
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(: (evolve (-> Number Number Number Number Number Number List)))
(= (evolve $pop_size $depth $tourn_size $cross_p $mut_p $gens)
  (evolve_loop (init_population $pop_size) $depth $tourn_size $cross_p $mut_p $gens))

(: (evolve_loop (-> List Number Number Number Number Number List)))
(= (evolve_loop $pop $depth $tourn_size $cross_p $mut_p $gen)
  (if (== $gen 0) 
        $pop
        (let* (($newpop (evolve_generation $pop $depth $tourn_size $cross_p $mut_p (size-atom $pop))))
              (evolve_loop $newpop $depth $tourn_size $cross_p $mut_p (- $gen 1)))))


(: (evolve_generation (-> List Number Number Number Number Number List)))
(= (evolve_generation $pop $depth $tourn_size $cross_p $mut_p $n)
  (if (== $n 0) 
      ()
    (let* (
            ($p1 (tournament_select $pop $tourn_size))
            ($p2 (tournament_select $pop $tourn_size))
            ($child (if (< (random-int &rng 0 10) (* $cross_p 10)) (crossover $p1 $p2) $p1))
            ($mutated (if (< (random-int &rng 0 10)(* $mut_p 10)) (mutate $child) $child))
         ) 
         ( 
          cons-atom $mutated (evolve_generation $pop $depth $tourn_size $cross_p $mut_p (- $n 1)) ))))

          
