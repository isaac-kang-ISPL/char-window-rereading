"""Character-window re-reading of beam hypotheses — the method of the LEGIT manuscript.

A scene-text recognizer (SVTRv2-AR) returns its greedy transcript a and a beam of hypotheses. Every hypothesis y in
the top K of the beam (K = 10; beam width 10, 10 expansions per hypothesis — the manuscript's stored runs: 20, 20) is
scored against a:

    S(y) = Δlog P_rec(y) + α·ΔPLL_ByT5(y) + β·Δlen(y) + γ·ΔW(y) − δ·ΔILM(y)      (Δ = value of y − value of a)

and the best hypothesis replaces a when its score exceeds τ. ΔW is the window term — the same recognizer re-reads
each character window of a with the substring of y that the window covers, and the window scores are summed with
coverage weights (`window`, `window_score`); PLL is the pseudo-log-likelihood of y under ByT5-XL (`lm`), its difference
taken as 0 when a or y has fewer than two characters (`decide.CONTEXT_MIN`); ILM is the internal language model of
the recognizer (`ilm`). Reading clusters (`decide.merge_readings`) are not part of the method; they remain for the
comparison rows of earlier results.

Coefficients are chosen on the validation set (natdev, 10,000 Union14M-L-val images) by the number of correct images
on one grid (`select.GRID`), and applied once to the evaluation sets (CB 6 sets, UB 7 sets). Accuracy is the macro
average over the sets of each benchmark, with the normalize_36_paper rule of the public SVTRv2 numbers (`metrics`).

Modules
    store        locations of the stored inputs (beam lists, score tables) under the data root
    hypotheses   the candidate rule: greedy transcript a and the beam top K, as Δ against a
    decide       the decision rule S(y) > τ, the window term (reading clusters for earlier results)
    select       the coefficient grid and the selection rule on the validation set
    metrics      CB / UB accuracy, paired comparisons, sign test
    ablation     rows that drop terms from the final combination (coefficients fixed)
    lm, ilm      ByT5 pseudo-log-likelihood, internal language model of the recognizer (GPU)
    nbest        beam search of SVTRv2-AR (GPU)
    position     attention maps → character points p_tl and the text-line curve tl_4
    segment      character segments (nearest seed; marker-controlled watershed for comparison)
    region       text-region band TR, the window region (band of 1.5 x-height; TRER with character boxes for comparison)
    window, window_score   character windows and their scores (GPU)
    cli          end-to-end entry point
"""
