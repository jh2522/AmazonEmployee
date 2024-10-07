library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(GGally)
library(skimr)
library(DataExplorer)
library(ggmosaic)
library(embed)
sample <- "sampleSubmission.csv"
test <- "test.csv"
train <- "train.csv"
sample1 <- vroom(sample)
test1 <- vroom(test)
train1 <- vroom(train)
train1
plot1 <- plot_correlation(train1)
plot1
plot2 <- glimpse(train1)
plot2

my_recipe <- recipe(ACTION~., data=train1) %>% 
  step_mutate_at(RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2, ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, ROLE_CODE, ACTION ,fn=factor) %>% 
  step_other(RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2, ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, ROLE_CODE, ACTION, threshold = .001) %>%
  step_dummy(RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2, ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, ROLE_CODE, ACTION)

prep <- prep(my_recipe)
baked <- bake(prep, new_data=train1)
baked
