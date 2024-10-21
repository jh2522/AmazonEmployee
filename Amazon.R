library(tidyverse)
library(tidymodels)
library(vroom)
library(ranger)
library(patchwork)
library(GGally)
library(skimr)
library(DataExplorer)
library(ggmosaic)
library(embed)
library(glmnet)
library(lme4)
library(kknn)

sample <- "sampleSubmission.csv"
test <- "test.csv"
train <- "train.csv"
sample1 <- vroom(sample)
test1 <- vroom(test)
train1 <- vroom(train)
train1
test1
plot1 <- plot_correlation(train1)
plot1
plot2 <- glimpse(train1)
plot2
mycleandata <- train1 %>% 
  mutate(ACTION=as.factor(ACTION))

my_recipe <- recipe(ACTION~., data=mycleandata) %>% 
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())


logRegModel <- logistic_reg() %>% 
  set_engine("glm")

logReg_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logRegModel) %>% 
  fit(data=mycleandata)

amazon_prediction <- predict(logReg_workflow,
                             new_data=test1,
                             type="prob")

kaggle_submission <- amazon_prediction %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1)

vroom_write(x=kaggle_submission, file="./Logistic12.csv", delim=",")

#penalized logistic regression

mycleandata <- train1 %>% 
  mutate(ACTION=as.factor(ACTION))

my_recipe <- recipe(ACTION~., data=mycleandata) %>% 
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  #step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% 
  #step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())


my_mod_pen <- logistic_reg(mixture=tune(), penalty=tune()) %>% 
  set_engine("glmnet")

amazon_workflow_pen <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod_pen)

tuning_grid_pen <- grid_regular(penalty(), mixture(), levels=7)

folds_pen <- vfold_cv(mycleandata, v = 5, repeats=1)

CV_results_pen <- amazon_workflow_pen %>% 
  tune_grid(resamples=folds_pen, grid=tuning_grid_pen, metrics=metric_set(roc_auc))
CV_results_pen
bestTune_pen <- CV_results_pen %>% 
  select_best(metric="roc_auc")
bestTune_pen
final_wf_pen <- 
  amazon_workflow_pen %>% 
  finalize_workflow(bestTune_pen) %>% 
  fit(data=mycleandata)

predict <- final_wf_pen %>% 
  predict(new_data=test1, type="prob")

kaggle_submission <- predict %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1)
vroom_write(x=kaggle_submission, file="./PenLogistic12.csv", delim=",")

#KNN

my_recipe <- recipe(ACTION~., data=mycleandata) %>% 
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())
knn_model <- nearest_neighbor(neighbors=50) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")
knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_model)



final_wf_knn <- 
  knn_wf %>% 
  fit(data=mycleandata)

predict_knn <- final_wf_knn %>% 
  predict(new_data=test1, type="prob")

kaggle_submission <- predict_knn %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1)
vroom_write(x=kaggle_submission, file="./KNNLogistic12.csv", delim=",")

#Random Forest

mycleandata <- train1 %>% 
  mutate(ACTION=as.factor(ACTION))

my_recipe <- recipe(ACTION~., data=mycleandata) %>% 
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())

forest_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

forest_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(forest_mod)

tuning_grid_forest <- grid_regular(mtry(range=c(1,9)), min_n(), levels=5)

folds_forest <- vfold_cv(mycleandata, v = 5, repeats=1)

CV_results_pen <- forest_wf %>% 
  tune_grid(resamples=folds_forest, grid=tuning_grid_forest, metrics=metric_set(roc_auc))
CV_results_pen
bestTune_pen <- CV_results_pen %>% 
  select_best(metric="roc_auc")