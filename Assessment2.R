# =====================================================================
# KB726 — Online Shoppers Intention 
# Dataset: online_shoppers.csv  
# Models: GLM (logistic), Random Forest, SVM (RBF), Neural Net 
# Notes:
# - RAM-friendly tuning grids (works better on ~3.6GB RAM my laptop has low ram so struggles to process)
# =====================================================================

# ----------------------------
# 0) Setup + packages
# ----------------------------
set.seed(123)

options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Required packages (prefer binaries to avoid Rtools issues)
pkgs_kb726 <- c(
  "tidyverse", "janitor",
  "tidymodels", "vip",
  "ranger", "kernlab",
  "nnet"
)

to_install_kb726 <- pkgs_kb726[!pkgs_kb726 %in% installed.packages()[, "Package"]]
if (length(to_install_kb726) > 0) install.packages(to_install_kb726, type = "binary")

library(tidyverse)
library(janitor)
library(tidymodels)
library(vip)

# ----------------------------
# 1) Load data
# ----------------------------
data_path_kb726 <- "online_shoppers.csv"
if (!file.exists(data_path_kb726)) stop("CSV not found: ", data_path_kb726)

raw_data_kb726 <- readr::read_csv(data_path_kb726, show_col_types = FALSE) |>
  clean_names()

cat("\n--- Data glimpse ---\n")
print(glimpse(raw_data_kb726))

cat("\n--- Target distribution (raw) ---\n")
print(raw_data_kb726 |> count(revenue) |> mutate(prop = n / sum(n)))

# Convert target to factor with positive class "TRUE"
df_kb726 <- raw_data_kb726 |>
  mutate(revenue = factor(revenue, levels = c(FALSE, TRUE), labels = c("FALSE", "TRUE")))

# Safety check
stopifnot(identical(levels(df_kb726$revenue), c("FALSE", "TRUE")))

# =====================================================================
# 2) EDA (Exploratory Data Analysis) — BEFORE splitting
# =====================================================================
cat("\n==================== EDA SECTION ====================\n")

# 2.1 Missingness check
missing_summary_kb726 <- df_kb726 |>
  summarise(across(everything(), ~ sum(is.na(.)))) |>
  pivot_longer(cols = everything(), names_to = "variable", values_to = "n_missing") |>
  arrange(desc(n_missing))

cat("\n--- Missing values per column ---\n")
print(missing_summary_kb726)

# 2.2 Class imbalance plot
class_counts_kb726 <- df_kb726 |>
  count(revenue) |>
  mutate(prop = n / sum(n))

cat("\n--- Class imbalance table ---\n")
print(class_counts_kb726)

p_class_kb726 <- ggplot(class_counts_kb726, aes(x = revenue, y = n)) +
  geom_col() +
  labs(
    title = "Class Imbalance: Revenue",
    x = "Revenue",
    y = "Count"
  ) +
  theme_minimal()

print(p_class_kb726)

# 2.3 Numeric summaries by class
# Identify numeric columns (excluding the target)
num_cols_kb726 <- df_kb726 |>
  select(where(is.numeric)) |>
  names()

numeric_summary_kb726 <- df_kb726 |>
  group_by(revenue) |>
  summarise(
    across(
      all_of(num_cols_kb726),
      list(mean = ~ mean(.x, na.rm = TRUE),
           sd   = ~ sd(.x, na.rm = TRUE),
           med  = ~ median(.x, na.rm = TRUE)),
      .names = "{.col}_{.fn}"
    ),
    .groups = "drop"
  )

cat("\n--- Numeric summary (mean/sd/median) by class ---\n")
print(numeric_summary_kb726)

# 2.4 Key predictor visual checks (quick but high-value)
# Page values vs revenue (often strong signal in this dataset)
p_pagevalues_kb726 <- df_kb726 |>
  ggplot(aes(x = revenue, y = page_values)) +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(
    title = "PageValues by Revenue",
    x = "Revenue",
    y = "PageValues"
  ) +
  theme_minimal()

print(p_pagevalues_kb726)

# Product-related duration vs revenue
p_prod_dur_kb726 <- df_kb726 |>
  ggplot(aes(x = revenue, y = product_related_duration)) +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(
    title = "ProductRelatedDuration by Revenue",
    x = "Revenue",
    y = "ProductRelatedDuration"
  ) +
  theme_minimal()

print(p_prod_dur_kb726)

# Month distribution by revenue (categorical)
p_month_kb726 <- df_kb726 |>
  count(month, revenue) |>
  group_by(month) |>
  mutate(prop = n / sum(n)) |>
  ungroup() |>
  ggplot(aes(x = month, y = prop, fill = revenue)) +
  geom_col(position = "stack") +
  labs(
    title = "Revenue Proportion by Month",
    x = "Month",
    y = "Within-month proportion"
  ) +
  theme_minimal()

print(p_month_kb726)

cat("\n==================== END EDA ========================\n")

# =====================================================================
# 3) Train / Test split
# =====================================================================
set.seed(123)
split_kb726 <- initial_split(df_kb726, prop = 0.80, strata = revenue)
train_kb726 <- training(split_kb726)
test_kb726  <- testing(split_kb726)

cat("\n--- Train/Test sizes ---\n")
print(c(train = nrow(train_kb726), test = nrow(test_kb726)))

cat("\n--- Target distribution (train) ---\n")
print(train_kb726 |> count(revenue) |> mutate(prop = n / sum(n)))

# =====================================================================
# 4) Recipe (preprocessing)
# =====================================================================
# Recipe:
# - Remove zero variance predictors
# - One-hot encode categorical predictors (month, visitor_type, weekend)
# - Normalize numeric predictors (helps GLM/SVM/NN)
recipe_kb726 <- recipe(revenue ~ ., data = train_kb726) |>
  step_zv(all_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# =====================================================================
# 5) Metrics + evaluation helper
# =====================================================================
# Probability metrics require: revenue + .pred_TRUE
metrics_prob_kb726 <- metric_set(
  yardstick::roc_auc,
  yardstick::pr_auc
)

# Class metrics require: revenue + .pred_class
metrics_class_kb726 <- metric_set(
  yardstick::accuracy,
  yardstick::f_meas,
  yardstick::sens,
  yardstick::specificity
)

# Cross-validation folds:
# - 5-fold is OK for GLM
set.seed(123)
folds_kb726 <- vfold_cv(train_kb726, v = 5, strata = revenue)

# Evaluate a fitted tidymodels workflow on test data:
evaluate_model_kb726 <- function(fitted_workflow_kb726, test_data_kb726) {
  
  probs_kb726 <- predict(fitted_workflow_kb726, test_data_kb726, type = "prob") |>
    bind_cols(test_data_kb726 |> dplyr::select(revenue))
  
  classes_kb726 <- predict(fitted_workflow_kb726, test_data_kb726, type = "class") |>
    bind_cols(test_data_kb726 |> dplyr::select(revenue))
  
  # IMPORTANT: TRUE is the positive class (second level)
  prob_metrics_kb726 <- metrics_prob_kb726(
    probs_kb726,
    truth = revenue,
    .pred_TRUE,
    event_level = "second"
  )
  
  class_metrics_kb726 <- metrics_class_kb726(
    classes_kb726,
    truth = revenue,
    estimate = .pred_class,
    event_level = "second"
  )
  
  list(
    metrics = bind_rows(prob_metrics_kb726, class_metrics_kb726),
    confusion = yardstick::conf_mat(classes_kb726, truth = revenue, estimate = .pred_class),
    probs = probs_kb726,
    classes = classes_kb726
  )
}


# =====================================================================
# 6) Model 1 — GLM (Logistic Regression)
# =====================================================================
cat("\n==================== Model 1: GLM (Logistic) ====================\n")

glm_spec_kb726 <- logistic_reg() |>
  set_engine("glm")

glm_workflow_kb726 <- workflow() |>
  add_model(glm_spec_kb726) |>
  add_recipe(recipe_kb726)

# CV (prob metrics)
set.seed(123)
glm_cv_kb726 <- fit_resamples(
  glm_workflow_kb726,
  resamples = folds_kb726,
  metrics = yardstick::metric_set(yardstick::roc_auc, yardstick::pr_auc)
)

cat("\n--- GLM CV metrics ---\n")
print(collect_metrics(glm_cv_kb726) |> arrange(desc(mean)))

# Final fit + test evaluation
glm_fit_kb726 <- fit(glm_workflow_kb726, data = train_kb726)
glm_eval_kb726 <- evaluate_model_kb726(glm_fit_kb726, test_kb726)

cat("\n--- GLM Test metrics ---\n")
print(glm_eval_kb726$metrics)

cat("\n--- GLM Confusion matrix ---\n")
print(glm_eval_kb726$confusion)

cat("\n--- GLM Top coefficients (abs) ---\n")
glm_coefs_kb726 <- glm_fit_kb726 |>
  extract_fit_parsnip() |>
  broom::tidy() |>
  arrange(desc(abs(estimate))) |>
  head(20)
print(glm_coefs_kb726)

# =====================================================================
# 7) Model 2 — Random Forest 
# =====================================================================
cat("\n==================== Model 2: Random Forest =======\n")

# Low-RAM tuning strategy:
# - 3-fold CV instead of 5
# - small space-filling grid (8 combos)
# - fewer trees during tuning (200)
# - single thread to avoid memory spikes
folds_rf_kb726 <- vfold_cv(train_kb726, v = 3, strata = revenue)

rf_spec_tune_kb726 <- rand_forest(
  trees = 200,
  mtry  = tune(),
  min_n = tune(),
  mode  = "classification"
) |>
  set_engine("ranger", importance = "impurity", num.threads = 1)

rf_workflow_kb726 <- workflow() |>
  add_model(rf_spec_tune_kb726) |>
  add_recipe(recipe_kb726)

rf_grid_kb726 <- grid_space_filling(
  mtry(range = c(3, 25)),
  min_n(range = c(2, 30)),
  size = 8
)

set.seed(123)
rf_tuned_kb726 <- tune_grid(
  rf_workflow_kb726,
  resamples = folds_rf_kb726,
  grid = rf_grid_kb726,
  metrics = yardstick::metric_set(yardstick::roc_auc, yardstick::pr_auc),
  control = control_grid(verbose = TRUE, save_pred = TRUE, allow_par = FALSE)
)

cat("\n--- RF CV metrics (top by ROC-AUC) ---\n")
print(
  collect_metrics(rf_tuned_kb726) |>
    filter(.metric %in% c("roc_auc", "pr_auc")) |>
    arrange(desc(mean))
)

best_rf_kb726 <- select_best(rf_tuned_kb726, metric = "roc_auc")
cat("\nBest RF hyperparameters:\n")
print(best_rf_kb726)

# Final RF with more trees (still safe)
rf_spec_final_kb726 <- rand_forest(
  trees = 500,
  mtry  = best_rf_kb726$mtry,
  min_n = best_rf_kb726$min_n,
  mode  = "classification"
) |>
  set_engine("ranger", importance = "impurity", num.threads = 1)

rf_final_wf_kb726 <- workflow() |>
  add_model(rf_spec_final_kb726) |>
  add_recipe(recipe_kb726)

rf_fit_kb726  <- fit(rf_final_wf_kb726, data = train_kb726)
rf_eval_kb726 <- evaluate_model_kb726(rf_fit_kb726, test_kb726)

cat("\n--- RF Test metrics ---\n")
print(rf_eval_kb726$metrics)

cat("\n--- RF Confusion matrix ---\n")
print(rf_eval_kb726$confusion)

cat("\n--- RF Variable Importance Plot ---\n")
try(vip(rf_fit_kb726, num_features = 15), silent = TRUE)

# =====================================================================
# 8) Model 3 — SVM  
# =====================================================================
cat("\n==================== Model 3: SVM  =============\n")

# Strategy (very fast due to low ram no space for big processing):
# - 2-fold CV
# - tiny grid (3 combos)
folds_svm_kb726 <- vfold_cv(train_kb726, v = 2, strata = revenue)

svm_spec_kb726 <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune(),
  mode = "classification"
) |>
  set_engine("kernlab")

svm_workflow_kb726 <- workflow() |>
  add_model(svm_spec_kb726) |>
  add_recipe(recipe_kb726)

svm_grid_kb726 <- grid_space_filling(
  cost(range = c(-2, 2)),
  rbf_sigma(range = c(-8, -2)),
  size = 3
)

set.seed(123)
svm_tuned_kb726 <- tune_grid(
  svm_workflow_kb726,
  resamples = folds_svm_kb726,
  grid = svm_grid_kb726,
  metrics = yardstick::metric_set(yardstick::roc_auc, yardstick::pr_auc),
  control = control_grid(verbose = TRUE, save_pred = TRUE, allow_par = FALSE)
)

cat("\n--- SVM CV metrics (top by ROC-AUC) ---\n")
print(
  collect_metrics(svm_tuned_kb726) |>
    filter(.metric %in% c("roc_auc", "pr_auc")) |>
    arrange(desc(mean))
)

best_svm_kb726 <- select_best(svm_tuned_kb726, metric = "roc_auc")
cat("\nBest SVM hyperparameters:\n")
print(best_svm_kb726)

svm_final_wf_kb726 <- finalize_workflow(svm_workflow_kb726, best_svm_kb726)
svm_fit_kb726 <- fit(svm_final_wf_kb726, data = train_kb726)
svm_eval_kb726 <- evaluate_model_kb726(svm_fit_kb726, test_kb726)

cat("\n--- SVM Test metrics ---\n")
print(svm_eval_kb726$metrics)

cat("\n--- SVM Confusion matrix ---\n")
print(svm_eval_kb726$confusion)

# =====================================================================
# 9) Model 4 — Neural Network 
# =====================================================================
cat("\n==================== Model 4: Neural Net (MLP via nnet) ============\n")

# Low-RAM tuning:
# - 3-fold CV
# - small grid (3 combos; keep it quick)
folds_nn_kb726 <- vfold_cv(train_kb726, v = 3, strata = revenue)

nn_spec_kb726 <- mlp(
  hidden_units = tune(),   # number of hidden neurons
  penalty = tune(),        # weight decay (regularisation)
  epochs = 200
) |>
  set_mode("classification") |>
  set_engine("nnet", MaxNWts = 20000, trace = FALSE)

nn_workflow_kb726 <- workflow() |>
  add_model(nn_spec_kb726) |>
  add_recipe(recipe_kb726)

# Small grid (fast)
nn_grid_kb726 <- tibble(
  hidden_units = c(5, 10, 15),
  penalty      = c(0.0001, 0.001, 0.01)
)

set.seed(123)
nn_tuned_kb726 <- tune_grid(
  nn_workflow_kb726,
  resamples = folds_nn_kb726,
  grid = nn_grid_kb726,
  metrics = yardstick::metric_set(yardstick::roc_auc, yardstick::pr_auc),
  control = control_grid(verbose = TRUE, save_pred = TRUE, allow_par = FALSE)
)

cat("\n--- NN CV metrics (top by ROC-AUC) ---\n")
print(
  collect_metrics(nn_tuned_kb726) |>
    filter(.metric %in% c("roc_auc", "pr_auc")) |>
    arrange(desc(mean))
)

best_nn_kb726 <- select_best(nn_tuned_kb726, metric = "roc_auc")
cat("\nBest NN hyperparameters:\n")
print(best_nn_kb726)

nn_final_wf_kb726 <- finalize_workflow(nn_workflow_kb726, best_nn_kb726)
nn_fit_kb726 <- fit(nn_final_wf_kb726, data = train_kb726)
nn_eval_kb726 <- evaluate_model_kb726(nn_fit_kb726, test_kb726)

cat("\n--- NN Test metrics ---\n")
print(nn_eval_kb726$metrics)

cat("\n--- NN Confusion matrix ---\n")
print(nn_eval_kb726$confusion)

# =====================================================================
# 10) Test set comparison table + ROC curves
# =====================================================================
cat("\n==================== Test Set Comparison ==========================\n")

glm_test_kb726 <- glm_eval_kb726$metrics |> mutate(model = "GLM")
rf_test_kb726  <- rf_eval_kb726$metrics  |> mutate(model = "Random Forest")
svm_test_kb726 <- svm_eval_kb726$metrics |> mutate(model = "SVM (RBF)")
nn_test_kb726  <- nn_eval_kb726$metrics  |> mutate(model = "Neural Net (MLP)")

comparison_kb726 <- bind_rows(glm_test_kb726, rf_test_kb726, svm_test_kb726, nn_test_kb726) |>
  select(model, .metric, .estimate) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

print(comparison_kb726)

cat("\n==================== ROC Curves (Test Set) ========================\n")

roc_data_kb726 <- bind_rows(
  glm_eval_kb726$probs |> mutate(model = "GLM") |> select(model, revenue, .pred_TRUE),
  rf_eval_kb726$probs  |> mutate(model = "Random Forest") |> select(model, revenue, .pred_TRUE),
  svm_eval_kb726$probs |> mutate(model = "SVM (RBF)") |> select(model, revenue, .pred_TRUE),
  nn_eval_kb726$probs  |> mutate(model = "Neural Net (MLP)") |> select(model, revenue, .pred_TRUE)
)

roc_df_kb726 <- roc_data_kb726 |>
  group_by(model) |>
  roc_curve(truth = revenue, .pred_TRUE)

roc_plot_kb726 <- ggplot(roc_df_kb726, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) +
  geom_abline(lty = 2) +
  labs(
    title = "ROC Curves (Test Set)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()

print(roc_plot_kb726)

# =====================================================================
# 11) Save key outputs (optional, helpful for report)
# =====================================================================
# Write comparison table to CSV for my report
write_csv(comparison_kb726, "model_comparison_kb726.csv")

# Save confusion matrices as text
sink("confusion_matrices_kb726.txt")
cat("=== GLM Confusion ===\n"); print(glm_eval_kb726$confusion)
cat("\n=== RF Confusion ===\n"); print(rf_eval_kb726$confusion)
cat("\n=== SVM Confusion ===\n"); print(svm_eval_kb726$confusion)
cat("\n=== NN Confusion ===\n"); print(nn_eval_kb726$confusion)
sink()

cat("\nDONE ✅ Outputs saved:\n- model_comparison_kb726.csv\n- confusion_matrices_kb726.txt\n")

