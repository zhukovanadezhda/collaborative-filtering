# Collaborative filtering for Recommendation Systems using Matrix Factorization

Imagine building a **recommendation system that predicts what movies you might like based on your past ratings and those of similar users**. Intuitively, this involves filling in a large matrix where rows represent users, columns represent movies, and the entries are ratings. However, most of this matrix is empty because users only rate a small fraction of movies. So how do we predict those missing ratings knowing that we have very little data for many users and items?

This is a classic problem in data science known as **collaborative filtering**. It looks for patterns in user behavior and item characteristics to infer preferences, even when direct ratings are sparse. This repository contains a project that implements this kind of model using **matrix factorization techniques**, to predict user-item ratings in a sparse matrix.

## 1. Problem Statement

We aim to predict unobserved user–item ratings in a sparse matrix $R \in \mathbb{R}^{m \times n}$, where each entry $R_{ui}$ is the rating given by user $u$ to item $i$.
Only a small subset of entries ($\Omega \subset [m] \times [n]$) is observed, making the task one of **matrix completion under sparsity**.

The model must generalize to unseen $(u, i)$ pairs, even when either the user or item has few interactions (the cold-start problem).

We measure predictive quality via the Root Mean Squared Error (RMSE):

$$
\mathrm{RMSE} =
\sqrt{
\frac{1}{|\Omega_{\mathrm{test}}|}
\sum_{(u,i) \in \Omega_{\mathrm{test}}}
(R_{ui} - \widehat{R}_{ui})^2
}.
$$


## 2. Our solution

Our final model represents the predicted rating $\widehat{R}_{ui}$ with several low-rank matrices $U, V, W_f$ learned from data, where $U$ and $V$ capture latent user and item factors, and $W_f$ project item features into the latent space. We further extend this approach by adding bias terms.

$$
\boxed{
\widehat{R}_{ui} =
U_u^\top \left(
V_i \times \sum_{f \in \mathcal{F}}
  W_f^\top x_{i,f}
  \right) + \mu + b_u + b_i
}
$$

where:

- $U \in \mathbb{R}^{m \times k}$: user latent factors
- $V \in \mathbb{R}^{n \times k}$: item latent factors
- $x_{i,f}$: feature vector for item (i) in feature group (f) (e.g. genre, year)
- $W_f \in \mathbb{R}^{d_f \times k}$: projection from feature space to latent space
- $\mu$: global mean;
- $b_u, b_i$: user and item biases
- $k$: latent dimension


<details markdown="1"> 
  <summary>📘 Details of different regularization terms used in the loss function (click to expand)</summary>

The loss minimized over observed ratings is enriched with several regularization terms to incorporate side information. Some of these regularizations are optional and can be deactivated, but the most complete formula of the loss function is:

$\qquad \qquad \qquad \large \mathcal{L} = \sum_{(u,i)\in \Omega} \bigl(R_{ui} - U_u^\top (V_i + \sum_f W_f^\top x_{i,f}) - \mu - b_u - b_i \bigr)^2 -$

$\qquad \qquad \qquad \large \textcolor{teal}{-\lambda_u \lVert U \rVert_F^2 - \sum_i \lambda_{v,i} \lVert V_i \rVert_2^2} \\ \textcolor{olive}{- \sum_f \lambda_{w_f} \lVert W_f \rVert_F^2} \\ \textcolor{brown}{- \lambda_{b_u} \lVert b_u \rVert_2^2 - \lambda_{b_i} \lVert b_i \rVert_2^2 -}$

$\qquad \qquad \qquad \large \textcolor{purple}{-\alpha\mathrm{Tr}(V^\top L V)},$

where:

- $\large \textcolor{teal}{\text{User or item shrinkage } (\lambda_u, \lambda_{v,i}):}$ This is the classic $L_2$ penalty that prevents latent factors from growing arbitrarily large. It ensures that users and items with few observations don’t overfit their small amount of data. In our case, the item regularization $\lambda_{v,i}$ can optionally be scaled inversely with the item’s popularity (we trust more the popular items): $\lambda_{v,i} = \frac{\lambda_v}{\sqrt{c_i + 1}},\quad c_i = |{u : (u,i)\in\Omega}|$.

- $\large \textcolor{olive}{\text{Feature projections } (\lambda_{w_f}):}$ Each $W_f$ projects an item’s raw feature vector (e.g. genre, year) into the same latent space as $V_i$. The corresponding regularization penalizes large deviations, acting like a prior that discourages the feature mappings from dominating the latent representation. Intuitively, it keeps the learned feature influence “gentle” rather than letting side information completely override collaborative patterns.

- $\large\textcolor{brown}{\text{Bias regularization } (\lambda_{b_u}, \lambda_{b_i}):}$ These terms keep user and item biases (i.e., consistent rating offsets) from absorbing too much variance. Without this constraint, biases could fit individual noise, especially for users who rate few items. In practice, bias regularization improves the stability and convergence speed of ALS updates.

- $\large \textcolor{purple}{\text{Item similarity } (\alpha \mathrm{Tr}(V^\top L V)):}$ This is the graph Laplacian regularization, which enforces that similar items (according to metadata) should have similar embeddings. It minimizes the smoothness term: $\mathrm{Tr}(V^\top L V) = \frac{1}{2} \sum_{i,j} S_{ij}|V_i - V_j|^2$, where $S$ encodes cosine similarity between items (e.g. shared genres). In practice, it acts like a “soft constraint” that pulls neighboring movies closer in latent space, which improves cold-item generalization when side metadata are available.

</details> 

Even though the global objective is non-convex, it becomes convex with respect to each parameter group when the others are fixed. This property allows an Alternating Least Squares (ALS) algorithm: at each step, one block of parameters is updated while keeping the rest constant. Each subproblem has a closed-form least-squares solution, which can be solved.

In simple terms:

- We alternate between updating users, items, feature projections, and biases.
- Each update “refits” that part of the model to the residuals left by the others.
- This process quickly drives the model toward a stationary point of the objective.

The updates repeat for a fixed number of iterations, but an early-stopping mechanism is applied to halt training once the validation RMSE stops improving, avoiding overfitting and saving computation.

<details markdown="1"> 
  <summary>🧮 Detailed update equations (click to expand)</summary>

At each iteration, we update each parameter in turn by solving the following subproblems (with Cholesky decomposition for efficiency):

$\qquad \large U_u \leftarrow \arg\min_x \lVert r_u - Z_u x \rVert_2^2 + \lambda_u \lVert x \rVert_2^2,$

$\qquad \large V_i \leftarrow \arg\min_x \lVert r_i - U_i x \rVert_2^2 + \lambda_{v,i}\lVert x \rVert_2^2 + \alpha \psi_i(x),$

$\qquad \large W_f \leftarrow \arg\min_x \sum_{(u,i)\in\Omega} (R_{ui} - U_u^\top (V_i + \sum_{f' \ne f} W_{f'}^\top x_{i,f'}) - \mu - b_u - b_i - U_u^\top x_{i,f} x) ^2 + \lambda_{w_f} \lVert x \rVert_F^2,$

$\qquad \large b_u \leftarrow \frac{\sum_{i: (u,i)\in\Omega} (R_{ui} - U_u^\top (V_i + \sum_f W_f^\top x_{i,f}) - \mu - b_i)} {\lVert \{i: (u,i)\in\Omega\}\rVert + \lambda_{b_u}},$

$\qquad \large b_i \leftarrow \frac{\sum_{u: (u,i)\in\Omega} (R_{ui} - U_u^\top (V_i + \sum_f W_f^\top x_{i,f}) - \mu - b_u)} {\lVert \{u: (u,i)\in\Omega\}\rVert + \lambda_{b_i}},$

$\qquad \large \mu \leftarrow \frac{1}{\lVert\Omega\rVert} \sum_{(u,i)\in\Omega} (R_{ui} - U_u^\top (V_i + \sum_f W_f^\top x_{i,f}) - b_u - b_i).$

Each term above corresponds to solving a small, independent least-squares problem. Since all updates have closed-form solutions, the method avoids costly gradient computations, making ALS both stable and interpretable.

</details> 


## 3. Hyperparameter Tuning

From our solution above, we can see that there are several hyperparameters to tune. So, once the model was implemented, we performed a hyperparameter search using Optuna with 3-fold cross-validation on the frozen folds that we created from the dataset. Each trial minimizes the mean validation RMSE across folds. We used a TPE sampler and MedianPruner, with early stopping enabled inside ALS (`es_tol = 1e−4`, `es_min_iters = 10`). The optimization ran for 150 trials.

**Best validation RMSE:** 0.8663  
**Mean training iterations:** 13 (all early-stopped folds)  
**Features used:** `genres`, `years`  
**Graph source:** `genres` (cosine similarity)  
**Regularization mode:** inverse-sqrt popularity scaling

> The best model combined feature-based item embeddings, graph regularization, and popularity-aware item shrinkage, suggesting that integrating side information improves generalization.

<details> 
  <summary>⚙️ The details on the hyperparameters and the used search space (click to expand)</summary>

| Hyperparameter           | Description                                    | Search Space               |
| ------------------------ | ---------------------------------------------- | -------------------------- |
| `n_factors`              | Latent dimensionality ($k$)                         | [1, 150]                   |
| `lambda_u`, `lambda_v`   | User/item $L_2$ regularization                    | [1e−4, 1e4] (log)          |
| `lambda_bu`, `lambda_bi` | Bias regularization                            | [1e−4, 1e4] (log)          |
| `lambda_w_<f>`           | Feature regularization (for `genres`, `years`) | [1e−4, 1e4] (log)          |
| `alpha`                  | Graph regularization strength                  | [0, 100]                   |
| `S_topk`                 | Graph similarity top-K                         | [1, 610]                   |
| `S_eps`                  | Graph epsilon cutoff                           | [1e−10, 1e−4]              |
| `pop_reg_mode`           | Item popularity weighting                      | {`None`, `inverse_sqrt`}   |
| `update_w_every`         | Frequency of W updates                         | [1, 60]                    |
| `n_iters`                | Max iterations                                 | 100 (fixed because of early-stopping) |

*Table 1. Hyperparameter search space used for Optuna optimization (150 trials, 3-fold CV).*

</details> 

## 4. Ablation Study

Since the best model combined all available components, to better understand the contribution of each of them to the overall performance, we performed an ablation study by disabling or altering one component at a time and measuring the impact on performance. All variants were evaluated on the same folds. The main metric is Test RMSE, complemented with per-popularity-bin RMSE to assess cold-item performance. We also report paired sign-test p-values against the baseline as well as FDR-corrected p-values to account for multiple comparisons. The variants tested are summarized in the table below:

| Variant                         | Description                                             | Disabled / Altered Components   |
| ------------------------------- | ------------------------------------------------------- | ------------------------------- |
| **full**               | All components enabled (baseline) | –                             |
| **no_features**        | Remove all side features          | $λ_{W_f}=0$             |
| **only_genres**        | Keep only genre features          | $λ_{W_{years}}=0$         |
| **only_years**         | Keep only year features           | $λ_{W_{genres}}=0$       |
| **no_graph**           | Disable Laplacian regularization  | $α=0$                           |
| **graph_feature=year** | Graph built from year similarity  | replace source feature for similarity computation        |
| **no_pop_reg**         | Uniform item regularization       | pop_reg_mode=None            |

*Table 2. Description of model variants tested in the ablation study and their altered components.*


<details> <summary>📊 Full results table (click to expand)</summary>

| Variant               | Time |Train RMSE| Test RMSE | Cold Bin RMSE | Popular Bin RMSE | Raw p-value | FDR Corrected p-value |
| --------------------- | ---- | -|-------- | -------------- | ----------------- | ------- | --------------------- |
| **full**|37.2270 $\pm$ 5.1053 |0.7900 $\pm$ 0.0017| **0.8618 $\pm$ 0.0069**|**0.9541 $\pm$ 0.0441**|**0.8495 $\pm$ 0.0043**| | |
| **no_features**|52.3489 $\pm$ 2.4499 |**0.7083 $\pm$ 0.0017**|1.0834 $\pm$ 0.0193|1.0602 $\pm$ 0.0784|1.1109 $\pm$ 0.0327| 0.0625| 0.1250|
| **only_genres**|36.1364 $\pm$ 2.2956 |0.7869 $\pm$ 0.0017|0.8714 $\pm$ 0.0059|0.9831 $\pm$ 0.0946|0.8602 $\pm$ 0.0057| 0.0625| 0.1250|
| **only_years**|59.9810 $\pm$ 3.4902 |0.7117 $\pm$ 0.0017|1.0034 $\pm$ 0.0087|1.0191 $\pm$ 0.0484|1.0135 $\pm$ 0.0125|0.0625| 0.1250|
| **no_graph**|**26.4977 $\pm$ 1.9122** |0.7900 $\pm$ 0.0017|**0.8618 $\pm$ 0.0069**|0.9541 $\pm$ 0.0441|**0.8495 $\pm$ 0.0043**| 1.000|1.000|
| **graph_feature=year**|42.0569 $\pm$ 1.1031 |0.7900 $\pm$ 0.0017|**0.8618 $\pm$ 0.0069**|0.9541 $\pm$ 0.0441|**0.8495 $\pm$ 0.0043**| 1.000|1.000|
| **no_pop_reg**|47.1860 $\pm$ 16.5324|0.7900 $\pm$ 0.0017|**0.8618 $\pm$ 0.0069**|0.9541 $\pm$ 0.0441|**0.8495 $\pm$ 0.0043**| 1.000|1.000|

*Table 3. Quantitative results of the ablation study. Reported metrics are mean ± std across folds. “Cold Bin RMSE” refers to items with ≤2 ratings; “Popular Bin RMSE” to items with 15+ ratings.*

</details>

## 5. Results and Discussion

The most visible effect appears when side features are removed: without genres and years, the test RMSE increases by about +0.22 (Fig.1 (a)), which suggests that these **metadata capture structure that the latent factors alone fail to model**. Using a single feature weakens performance: `only_genres` stays close, `only_years` drops more, meaning that **the two features look complementary rather than interchangeable**. Disabling the graph term (`no_graph`) or switching its similarity source (`graph_feature=years`) produces almost no change in RMSE, but consistently shortens training (Fig.1 (b)), because the construction of the similarity graph takes time with no visible impact on the final performance. The removing popularity-aware shrinkage (`no_pop_reg`) does not affect RMSE much either, though the idea of popularity-awareness could be further explored (here we only tested uniform vs. inverse-sqrt scaling).

Overall, side features seem to contribute most to predictive accuracy. Looking at per-popularity-bin RMSE (Fig.1 (c)), we see that RMSE mainly drops when years and genres are added, but the other components do not seem to affect the performance even in the cold bin (1–2 ratings).

However, even this small effect does not pass the significance threshold ($\alpha = 0.05$ after FDR correction), so the observations above should be interpreted as simple tendencies rather than definitive conclusions. Increasing the number of folds or seeds or implementing nested cross-validation would likely improve statistical power and make these patterns clearer.

<p align="center">
  <img src="assets/rmse_bar.png" width="29%" />
  <img src="assets/time_bar.png" width="29%" />
  <img src="assets/bins_grouped_bars.png" width="37.5%" />
</p>
<p align="center">
  <em>Fig. 1: (a) Test RMSE by variant; (b) Training time by variant; (c) Test RMSE per item-popularity bin (mean ± std across folds).</em>
</p>

## 6. Conclusion

This project implemented a recommender system based on matrix factorization, extended with graph regularization, side-feature projections, and popularity-dependent item shrinkage. The results suggest that feature-based item embeddings contribute most to predictive quality, while graph and popularity terms primarily help with stability and calibration.

Although the method itself is simple, particular care was taken to implement a reliable workflow for model selection, hyperparameter tuning, and ablation analysis. The code was structured with reusability in mind: it should be straightforward to add new features, modify the similarity graph, or test temporal extensions. The project can therefore serve as a solid baseline that can be reused and expanded to serve for more complex models in future work.

## References
[1] M. Belkin and P. Niyogi, “Laplacian eigenmaps and spectral techniques for embedding and clustering,” in Advances in Neural Information Processing Systems (NeurIPS), MIT Press, 2001, pp. 585–591.

[2] Y. Li, J. Lee, and Y. Lee, “Matrix factorization with graph regularization for collaborative filtering,” Proceedings of the 24th ACM international conference on information and knowledge management, pp. 79–88, 2015.
