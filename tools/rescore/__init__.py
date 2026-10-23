"""Character-window re-reading of beam hypotheses — the method of the CWR manuscript.

A scene-text recognizer (SVTRv2-AR) returns its greedy transcript a and a beam of hypotheses. Every hypothesis y in
the top K of the beam (K = 10; beam width 10, 10 characters expanded per hypothesis) is scored against a:

    S(y) = Δb(y) + α·ΔV(y) + β·(|y| − |a|) + γ·ΔPLL(y)           (Δ = value of y − value of a)

and the best hypothesis replaces a when its score exceeds τ. Δb is the recognizer's log-probability difference. ΔV is
the window term — the same recognizer re-reads each character window W_w of a (MW3 midpoint windows, `midpoint`) with
the substring of y that the window covers, and the window scores are summed with the overlap weights ω_w
(`window_score`). PLL is the pseudo-log-likelihood of y under ByT5-XL (`lm`), its difference taken as 0 when a or y has
fewer than two characters (`decide.CONTEXT_MIN`). The image-conditioned extension (`condition`) changes only α, β and
γ per image. Reading clusters (`decide.merge_readings`) and the internal-LM rows are not part of the method; they
remain for comparison rows of earlier results.

Paper ↔ code: α = `omega`, β = `beta`, γ = `mu`, τ = `tau`; V(y) is called W in the window tables, ω_w is `u_w`.

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
    condition    the image-conditioned extension (judge answers → coefficients per image)
    lm           ByT5 pseudo-log-likelihood (GPU)
    nbest        beam search of SVTRv2-AR (GPU)
    position     attention maps → character points p_i and the text-line curve
    region       text region R (band of 1.5 x-height around the text line)
    midpoint     MW1–MW9 midpoint windows (MW3 is the W3 of the paper)
    segment, window   character segments and windows of an earlier window definition (`--window w3`; not used in
                 the paper); window crops and inputs shared with the midpoint windows
    window_score window scores and their overlap-weighted sum V(y) (GPU)
    pipeline, cli   end-to-end runs and the command-line entry point
"""
