library(tidyverse)
library(afex)
library(ggplot2)
library(emmeans)
library(lmtest)

tmp.d <-
  read.csv("~/tmp.csv")

str(tmp.d)

m1 <-
  tmp.d %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1)) %>%
  filter(NbObjects==1) %>%
  mixed(RT ~ Colour1 + (1|Subject), data=., method='S')

emmeans(m1, "Colour1")

m2 <-
  tmp.d %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Matching = factor(Matching)) %>%
  filter(NbObjects==1) %>%
  droplevels %>%
  mixed(RT ~ Matching + (1|Subject), data=., method='S')
emmeans(m2, "Matching")

tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Matching = factor(Matching),
         Difficulty_coded = case_when(Difficulty=="D0"~-1,Difficulty=="D1"~0,Difficulty=="D2"~1,)) %>%
  droplevels %>%
  mixed(RT ~ Difficulty_coded+ (1|Subject), data=., method='S') %>%
  summary

m3 <-
  tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Matching = factor(Matching),
         D0 = Difficulty == "D0",
         D1 = Difficulty == "D1",
         D2 = Difficulty == "D2",
         Difficulty = factor(Difficulty, levels=c("D0", "D1", "D2"))) %>%
  droplevels %>%
  mixed(RT ~ Difficulty + (1|Subject), data=., method='S')

emm3 <- emmeans(m3, "Difficulty", lmerTest.limit = 8240)

des_c <- list(
  D0D1 = c(1, -1, 0),
  D0D2 = c(1, 0, -1),
  D1D2 = c(0, 1, -1))
contrast(emm3, des_c)

m4 <-
  tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Matching = factor(Matching),
         D0 = Difficulty == "D0",
         D1 = Difficulty == "D1",
         D2 = Difficulty == "D2",
         Difficulty = factor(Difficulty, levels=c("D0", "D1", "D2"))) %>%
  droplevels %>%
  mixed(Perf ~ Difficulty + (1|Subject), data=., family="binomial", method='LRT')

emm4 <- emmeans(m4, "Difficulty", lmerTest.limit = 8240)

des_c4 <- list(
  D0D1 = c(1, -1, 0),
  D0D2 = c(1, 0, -1),
  D1D2 = c(0, 1, -1))
contrast(emm4, des_c4)

tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Matching = factor(Matching),
         Difficulty = factor(Difficulty),
         Difficulty_coded = case_when(Difficulty=="D0"~-1,Difficulty=="D1"~0,Difficulty=="D2"~1,)) %>%
  droplevels %>%
  group_by(Difficulty, Subject) %>%
  summarize(RT=mean(Perf)) %>%
  ggplot(aes(x=Difficulty, y=RT, color=Subject, group=Subject)) +
  geom_point() +
  geom_line()

tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Matching = factor(Matching),
         Difficulty = factor(Difficulty),
         Difficulty_coded = case_when(Difficulty=="D0"~-1,Difficulty=="D1"~0,Difficulty=="D2"~1,)) %>%
  droplevels %>%
  group_by(Difficulty, Subject) %>%
  summarize(RT=mean(RT)) %>%
  ggplot(aes(x=Difficulty, y=RT, color=Subject, group=Subject)) +
  geom_point() +
  geom_line()



m5 <-
  tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Colour2 = factor(Colour2),
         Shape1 = factor(Shape1),
         Shape2 = factor(Shape2),
         Matching = factor(Matching),
         Difficulty = factor(Difficulty)) %>%
  droplevels %>%
  mixed(RT ~ Difficulty + Colour1 * Shape1 + Colour2 * Shape2 + (1|Subject), data=., method='S')

emmeans(m5, "Shape1")

m6 <-
  tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Colour2 = factor(Colour2),
         Shape1 = factor(Shape1),
         Shape2 = factor(Shape2),
         Matching = factor(Matching),
         Difficulty = factor(Difficulty)) %>%
  droplevels %>%
  mixed(RT ~ Difficulty + (1|Subject), data=., method='S')

summary(m6)
emmeans(m6, "Difficulty", lmerTest.limit=8240, pbkrtest.limit=8240)

AIC(m5$full_model, m6$full_model)
lrtest(m5$full_model, m6$full_model)

tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Matching = factor(Matching),
         Difficulty = factor(Difficulty),
         Difficulty_coded = case_when(Difficulty=="D0"~-1,Difficulty=="D1"~0,Difficulty=="D2"~1,)) %>%
  droplevels %>%
  group_by(Shape1, Subject) %>%
  summarize(RT=mean(RT)) %>%
  ggplot(aes(x=Shape1, y=RT, color=Subject, group=Subject)) +
  geom_point() +
  geom_line()


m7 <-
  tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Colour2 = factor(Colour2),
         Shape1 = factor(Shape1),
         Shape2 = factor(Shape2),
         Matching = factor(Matching),
         Difficulty = factor(Difficulty)) %>%
  droplevels %>%
  mixed(Perf ~ Difficulty + Colour1 * Shape1 + Colour2 * Shape2 + (1|Subject), data=., method='LRT', family="binomial")

summary(m7)
contrast(emmeans(m7, c("Colour2","Shape2")))


m8 <-
  tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Colour2 = factor(Colour2),
         Shape1 = factor(Shape1),
         Shape2 = factor(Shape2),
         Matching = factor(Matching),
         Error_type = factor(Error_type),
         Difficulty = factor(Difficulty)) %>%
  droplevels %>%
  mixed(Perf ~ Difficulty + Difficulty:Error_type + (1|Subject), data=., method='LRT', family="binomial")

nice(m8)
summary(m8)
contrast(emmeans(m8, c("Diffculty:Error_type")))


# Difficulty + Error_type RT
m9 <-
  tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Colour1 = factor(Colour1),
         Colour2 = factor(Colour2),
         Shape1 = factor(Shape1),
         Shape2 = factor(Shape2),
         Matching = factor(Matching),
         Error_type = factor(Error_type),
         Difficulty = factor(Difficulty)) %>%
  droplevels %>%
  mixed(RT ~ Difficulty + Difficulty:Error_type + (1|Subject), data=., method='S')

summary(m9)
# emmeans(m9, c("Difficulty","Error_type"))
# des_c <- list(D1L2D2L2 = c(0, 0, 0,0,0,0,0,-1,1,0,0,0),
#               D2L0D2L1 = c(0, 0, -1,0,0,1,0,0,0,0,0,0),
#               D2ND2L2 = c(0, 0, 0,0,0,0,0,0,-1,0,0,1)
#               )
# contrast(emmeans(m9, c("Difficulty","Error_type")), des_c)

m10 <-
  tmp.d %>% 
  filter(NbObjects==2) %>%
  mutate(Subject = factor(Subject),
         Matching = factor(Matching),
         Error_type = factor(Error_type),
         Difficulty = factor(Difficulty)) %>%
  droplevels %>%
  mixed(Perf ~ Difficulty * Error_type + (1|Subject), data=., method='LRT', family="binomial")


summary(m10)
emmeans(m10, c("Difficulty","Error_type"))
des_c <- list(D1L2D2L2 = c(0, 0, 0,0,0,0,0,-1,1,0,0,0),
              D2L0D2L1 = c(0, 0, -1,0,0,1,0,0,0,0,0,0),
              D2ND2L2 = c(0, 0, 0,0,0,0,0,0,-1,0,0,1)
              )
contrast(emmeans(m9, c("Difficulty","Error_type")), des_c)
