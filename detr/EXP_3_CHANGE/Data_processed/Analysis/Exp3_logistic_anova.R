# libraries
if(!require(dplyr)) install.packages("dplyr")
if(!require(devtools)) install.packages("devtools")
library("ggpubr")
devtools::install_github("kassambara/ggpubr")
set.seed(1234)

# load data, modify the path if necessary
# setwd("xxxxx")
exp_data <- read.csv('exp3a_data.csv')
control_data <- read.csv('exp3b_data.csv')
my_data <- rbind(exp_data, control_data)
dim(exp_data)
dim(control_data)
dim(my_data)
str(my_data)

# sample and show some example data 
dplyr::sample_n(my_data, 10)

# redefine data type and add columns
my_data$response[my_data$response== "same"] <- 0
my_data$response[my_data$response== "different"] <- 1
my_data$changeType[grepl('concave_area', my_data$changeType)] <- 'Concave'
my_data$changeType[grepl('concave_nofill', my_data$changeType)] <- 'NoFill'
my_data$changeType[grepl('convex_area', my_data$changeType)] <- 'Convex'
my_data$experimentVersion[grepl('v001', my_data$experimentVersion)] <- '3a'
my_data$experimentVersion[grepl('control', my_data$experimentVersion)] <- '3b'

# remove no_change rows and catch_shape rows
my_data <- my_data[!grepl("no_change", my_data$changeType),]
my_data <- my_data[!grepl("catch_shape", my_data$shape),]

# change data type
my_data$Response <- as.numeric(my_data$response)
my_data$ExperimentVersion <- factor(my_data$experimentVersion,
                                    levels=c("3a", "3b"),
                                    labels=c("3a", "3b"))
my_data$ChangeType <- factor(my_data$changeType,
                             levels=c("Concave", "NoFill", "Convex"),
                             labels=c("Concave", "NoFill", "Convex"))

# select three columns for analysis
my_data <- my_data[, c("ChangeType", "ExperimentVersion", "Response")]
dim(my_data)
str(my_data)
table(my_data$ChangeType, my_data$ExperimentVersion) # frequency
summary(my_data)

# simply visualize raw data
ggline(my_data, x = "ChangeType", y = "Response", 
       color = "ExperimentVersion",
       add = c("mean_se"),
       palette = c("#00AFBB", "#E7B800"),
       ylim=c(0,1.0),
       xlab="Change Type",
       ylab="%Change")

# two way anova (binomial)
binomial <- glm(Response ~ ChangeType * ExperimentVersion, 
                data = my_data, 
                family=binomial(link="logit"))
summary(binomial)
binomial
plot(binomial, 1)
binomial.anova = anova(binomial, test='Chisq') # chi squared test
binomial.anova 
summary(binomial.anova)

