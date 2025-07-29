!(register-module! ../GGGP)
!(import! &self GGGP:GGGP)



; ! "Test: create_individual"
; !(create_individual expr 0)

; ! "Test: contains_symbol"
; !(contains_symbol (expr var) expr) ;;True
; ; !(contains_symbol (const 1) expr)   ;; False

! "Test: element_at 2 index"
!(element_at (10 20 30 40) 2)

! "Random Selection from given list"
!(select_random (10 20 30 40 50))             

;; test case and expected output for is_list 
;; True
! "Print True if it Listed in Atomspace, otherway Fasle:"
! (is_list (expr (expr op expr))) ;; expected True


; ! "Test: filter_terminals"
; !(filter_terminals ((expr const) (expr (expr op expr)) (expr var))) ;; True 1 itr
; !(filter_terminals ((expr (expr op expr)) (expr var))) ;;2nd itr true
; !(filter_terminals ((expr var))) 
