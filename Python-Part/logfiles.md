# Logs for python experiemnts

## Jun 5, 2023
Moved to desk, but remember this is for resource competition project.

## Jun 6, 2023
Try to write a code to sample $G$, $C$, $R^*$ and $S^*$. Check how large the files are. And use pytorch to do optimization, linear algebra. Use MATLAB to do all simulations. At last, draw figures with python? 

## Jun 7, 2023
Tried neural ode, too slow. Shift to julia. Save all the documents in .mat.

## Jun 10, 2023
Aternative stable state fraction not normal. Guess PCA does not work when the basin is large enough. Try to use the conservative criterion: whether the same set of species survive.

## Jun 11, 2023
Write .py for brutally check alternative stable states.

## Jun 16, 2023
Draw fluctuation fraction modefied by real Jacobian. Brute force, first try $N_S = N_R$ linear system, then $N_S \neq N_R$ nonlieanr system.

## Jun 18, 2023
Too many things this weekend. Let's wrap up what have been done, what new understanding we got, and what the next steps are (soome data got, have not analyze).

After a long time of thinking, the plan is as follows
* Fig 2: still metabolic trade-off, but only with one figure of unstable fraction (phase diagram or just dots)

* Fig 3: general encroachment $E(G,C)$, with figures about unstable fractions for three models, we have got all the things. 

* Fig 4: main text focuses on $N_S = N_R$, surviving fractions, flucturation fractions, alternative stable states fractions (waiting for simulation results from logistic resource supply), and number of alternative stable states (got data, have not analyzed).

* Fig 5: phase diagrams for simulation reults in Fig 4.

## Jun 16
Try to do more detailed tests on dynamics. Fractions of chaos, limit cycles, steady stable, and steady unstable.

## Jun 25
In solving alternative stable states, initial conditions seem to be important. Changed initial sampling $\mathcal{U}(0,100)$ from $\mathcal{U}(0,1)$. Hopefully, we can find more AltSS for logistic resource supply. Pay attention to distinguish new and old data after getting new. 

## Jul 5
Are the alternative stable states before losing local stability real or not? Try to make the criterion lower (from $10^{-3}$ to $10^{-6}$ in brute-force solution) to check.