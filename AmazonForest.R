library(tidyverse)
library(tidymodels)
library(vroom)
library(ranger)
library(embed)

sample <- "sampleSubmission.csv"
test <- "test.csv"
train <- "train.csv"
sample1 <- vroom(sample)
test1 <- vroom(test)
train1 <- vroom(train)

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

tuning_grid_forest <- grid_regular(mtry(range=c(1,9)), min_n(), levels=8)

folds_forest <- vfold_cv(mycleandata, v = 5, repeats=1)

CV_results_forest <- forest_wf %>% 
  tune_grid(resamples=folds_forest, grid=tuning_grid_forest, metrics=metric_set(roc_auc))

bestTune_forest <- CV_results_forest %>% 
  select_best(metric="roc_auc")

final_wf_forest <- 
  forest_wf %>% 
  finalize_workflow(bestTune_forest) %>% 
  fit(data=mycleandata)

predict <- final_wf_forest %>% 
  predict(new_data=test1, type="prob")

kaggle_submission <- predict %>% 
  bind_cols(., test1) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION=.pred_1)
vroom_write(x=kaggle_submission, file="./ForestLogistic12.csv", delim=",")
