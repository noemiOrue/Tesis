setwd("C:/Users/bcoma/Documents/GitHub/Tesis_UB/scripts")
library(readxl)
#install.packages("ltm")
#install.packages("polycor")
#install.packages("glmmTMB")
library(polycor)
library("glmmTMB")


#cor <- correlation(allInfo)
#summary(cor)

allInfo <- read_excel('../output/allExcels_negatiu.xlsx')

model1 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfo)
summary(model1)


model1log <- glm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfo,family = Gamma(link=inverse))
summary(model1log)
with(summary(model1log), 1 - deviance/null.deviance)

model1logI <- glmmTMB(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfo,family = Gamma(link=inverse))
summary(model1logI)

modelbin1 <- glm(Visitado~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfo,family = "binomial")
summary(modelbin1)
nullmod <- glm(Visitado~1,data=allInfo, family="binomial")
1-logLik(modelbin1)/logLik(nullmod)



allInfoLog <- allInfo[,]

allInfoLog['Dinero_en_el_proyecto'] <- log2(allInfoLog['Dinero_en_el_proyecto']) 
allInfoLog['NGO_Country_Budget_Previous_Year'] <- log2(allInfoLog['NGO_Country_Budget_Previous_Year'])
allInfoLog['Total_subvencion_en_el_Pais_y_Anyo'] <- log2(allInfoLog['Total_subvencion_en_el_Pais_y_Anyo'])
allInfoLog['Gross_National_Income'] <- log2(allInfoLog['Gross_National_Income'])
allInfoLog['Public_Grant'] <- log2(allInfoLog['Public_Grant'])

allInfoLog[allInfoLog<0] <- 0


model1 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog)
summary(model1)


model1log <- glm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog,family = quasipoisson(link = "log"))
summary(model1log)
with(summary(model1log), 1 - deviance/null.deviance)

modelbin1 <- glm(Visitado~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog,family = "binomial")
summary(modelbin1)
nullmod <- glm(Visitado~1,data=allInfoLog, family="binomial")
1-logLik(modelbin1)/logLik(nullmod)



















cor.test(allInfo$NGO_Country_Budget_Previous_Year,allInfo$Delegacion, method = "pearson")
#biserial.cor(allInfo$NGO_Country_Budget_Previous_Year,allInfo$Delegacion)
polyserial(allInfo$NGO_Country_Budget_Previous_Year,allInfo$Delegacion)

cor.test(allInfo$ONU,allInfo$Vision_ONGD_Africa, method = "pearson")


cor.test(allInfo$NGO_Country_Budget_Previous_Year,allInfo$Dinero_en_el_proyecto, method = "pearson")
#cor.test(log(allInfo$NGO_Country_Budget_Previous_Year),log(allInfo$Dinero_en_el_proyecto), method = "pearson")


cor.test(allInfo$ONU,allInfo$Gross_National_Income, method = "pearson")
cor.test(allInfo$ONU,allInfo$Public_Grant, method = "pearson")
cor.test(allInfo$ONU,allInfo$Proporcion_Fondos_Privados, method = "pearson")
cor.test(allInfo$ONU,allInfo$dinero_anyo_anterior_en_proyectos, method = "pearson")
cor.test(allInfo$ONU,allInfo$Total_subvencion_en_el_Pais_y_Anyo, method = "pearson")
cor.test(allInfo$ONU,allInfo$Vision_ONGD_Latinoamerica , method = "pearson")
cor.test(allInfo$ONU,allInfo$Vision_ONGD_Africa , method = "pearson")
cor.test(allInfo$ONU,allInfo$Vision_ONGD_Universal , method = "pearson")

cor.test(allInfo$Gross_National_Income,allInfo$Subvencion_publica, method = "pearson")
cor.test(allInfo$Gross_National_Income,allInfo$Fondos_Publicos_MAE, method = "pearson")
cor.test(allInfo$Gross_National_Income,allInfo$Proporcion_Fondos_Privados, method = "pearson")
cor.test(allInfo$Gross_National_Income,allInfo$dinero_anyo_anterior_en_proyectos, method = "pearson")
cor.test(allInfo$Gross_National_Income,allInfo$Total_subvencion_en_el_Pais_y_Anyo, method = "pearson")
cor.test(allInfo$Gross_National_Income,allInfo$Vision_ONGD_Latinoamerica, method = "pearson")
cor.test(allInfo$Gross_National_Income,allInfo$Vision_ONGD_Africa, method = "pearson")
cor.test(allInfo$Gross_National_Income,allInfo$Vision_ONGD_Universal, method = "pearson")

cor.test(allInfo$Subvencion_publica,allInfo$Fondos_Publicos_MAE, method = "pearson")
cor.test(allInfo$Subvencion_publica,allInfo$Proporcion_Fondos_Privados, method = "pearson")
cor.test(allInfo$Subvencion_publica,allInfo$dinero_anyo_anterior_en_proyectos, method = "pearson")
cor.test(allInfo$Subvencion_publica,allInfo$Total_subvencion_en_el_Pais_y_Anyo, method = "pearson")
cor.test(allInfo$Subvencion_publica,allInfo$Vision_ONGD_Latinoamerica, method = "pearson")
cor.test(allInfo$Subvencion_publica,allInfo$Vision_ONGD_Africa, method = "pearson")
cor.test(allInfo$Subvencion_publica,allInfo$Vision_ONGD_Universal, method = "pearson")

cor.test(allInfo$Fondos_Publicos_MAE,allInfo$Proporcion_Fondos_Privados, method = "pearson")
cor.test(allInfo$Fondos_Publicos_MAE,allInfo$dinero_anyo_anterior_en_proyectos, method = "pearson")
cor.test(allInfo$Fondos_Publicos_MAE,allInfo$Total_subvencion_en_el_Pais_y_Anyo, method = "pearson")
cor.test(allInfo$Fondos_Publicos_MAE,allInfo$Vision_ONGD_Latinoamerica, method = "pearson")
cor.test(allInfo$Fondos_Publicos_MAE,allInfo$Vision_ONGD_Africa, method = "pearson")
cor.test(allInfo$Fondos_Publicos_MAE,allInfo$Vision_ONGD_Universal, method = "pearson")

cor.test(allInfo$Proporcion_Fondos_Privados,allInfo$dinero_anyo_anterior_en_proyectos, method = "pearson")
cor.test(allInfo$Proporcion_Fondos_Privados,allInfo$Total_subvencion_en_el_Pais_y_Anyo, method = "pearson")
cor.test(allInfo$Proporcion_Fondos_Privados,allInfo$Vision_ONGD_Latinoamerica, method = "pearson")
cor.test(allInfo$Proporcion_Fondos_Privados,allInfo$Vision_ONGD_Africa, method = "pearson")
cor.test(allInfo$Proporcion_Fondos_Privados,allInfo$Vision_ONGD_Universal, method = "pearson")

cor.test(allInfo$dinero_anyo_anterior_en_proyectos,allInfo$Total_subvencion_en_el_Pais_y_Anyo, method = "pearson")
cor.test(allInfo$dinero_anyo_anterior_en_proyectos,allInfo$Vision_ONGD_Latinoamerica, method = "pearson")
cor.test(allInfo$dinero_anyo_anterior_en_proyectos,allInfo$Vision_ONGD_Africa, method = "pearson")
cor.test(allInfo$dinero_anyo_anterior_en_proyectos,allInfo$Vision_ONGD_Universal, method = "pearson")


cor.test(allInfo$Total_subvencion_en_el_Pais_y_Anyo,allInfo$Vision_ONGD_Latinoamerica, method = "pearson")
cor.test(allInfo$Total_subvencion_en_el_Pais_y_Anyo,allInfo$Vision_ONGD_Africa, method = "pearson")
cor.test(allInfo$Total_subvencion_en_el_Pais_y_Anyo,allInfo$Vision_ONGD_Universal, method = "pearson")

cor.test(allInfo$Vision_ONGD_Latinoamerica,allInfo$Vision_ONGD_Africa, method = "pearson")
cor.test(allInfo$Vision_ONGD_Latinoamerica,allInfo$Vision_ONGD_Universal, method = "pearson")

hist(allInfo$Fondos_Publicos_MAE)
hist(allInfo$Dinero_en_el_proyecto)
plot(allInfo$Dinero_en_el_proyecto,dnorm(allInfo$Dinero_en_el_proyecto))

#allInfo["Anyo_ONG"]= 2020-allInfo["Anyo_ONG"]
allInfo

model1 <- lm(Dinero_en_el_proyecto~Gross_National_Income+Public_Grant+Total_Fondos+NGO_Country_Budget_Previous_Year+Internacional+Colony+Delegacion,data=allInfo)
summary(model1)



modelbin1 <- lm(Dinero_en_el_proyecto~Gross_National_Income+Public_Grant+Total_Fondos+NGO_Country_Budget_Previous_Year+Internacional+Colony+Delegacion,data=allInfo)
summary(model1)






#hist(allInfo$Dinero_en_el_proyecto)
#hist(allInfoLog$Dinero_en_el_proyecto)

model <- glm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog,Gamma(link = "log"))
summary(model)




model <- glm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog,family = "quasipoisson")
summary(model)

error = rstandard(model)

qqnorm(error)
qqline(error)

library(olsrr)


d<-density(model[['residuals']])
plot(d,main='Residual KDE Plot',xlab='Residual value')

ols_plot_resid_qq(model)
ols_plot_resid_qq(model6)


plot(model)


model3 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog)
summary(model3)

model4 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony,data=allInfoLog)
summary(model4)

model5 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony,data=allInfoLog)
summary(model5)

model6 <- lm(Dinero_en_el_proyecto~NGO_Country_Budget_Previous_Year+Delegacion,data=allInfoLog)
summary(model6)



#model1gamma <- glm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog,family  = Gamma(link = "log"))
#summary(model1gamma)

modelbin1 <- glm(Visitado~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog,family = "binomial")
summary(modelbin1)
nullmod <- glm(Visitado~1,data=allInfoLog, family="binomial")
1-logLik(modelbin1)/logLik(nullmod)


modelbin2 <- glm(Visitado~Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_Universal+Colony+Delegacion,data=allInfoLog,family = "binomial")
summary(modelbin2)
nullmod <- glm(Visitado~1,data=allInfoLog, family="binomial")
1-logLik(modelbin2)/logLik(nullmod)

modelbin3 <- glm(Visitado~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog,family = "binomial")
summary(modelbin3)
nullmod <- glm(Visitado~1,data=allInfoLog, family="binomial")
1-logLik(modelbin3)/logLik(nullmod)

modelbin4 <- glm(Visitado~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony,data=allInfoLog,family = "binomial")
summary(modelbin4)
nullmod <- glm(Visitado~1,data=allInfoLog, family="binomial")
1-logLik(modelbin4)/logLik(nullmod)

modelbin5 <- glm(Visitado~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony,data=allInfoLog,family = "binomial")
summary(modelbin5)
nullmod <- glm(Visitado~1,data=allInfoLog, family="binomial")
1-logLik(modelbin5)/logLik(nullmod)


modelbin6 <- glm(Visitado~NGO_Country_Budget_Previous_Year+Delegacion,data=allInfoLog,family = "binomial")
summary(modelbin6)
nullmod <- glm(Visitado~1,data=allInfoLog, family="binomial")
1-logLik(modelbin6)/logLik(nullmod)






model2 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+Colony+Delegacion,data=allInfoLog)
summary(model2)


model3 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Public_Grant+Total_Fondos+Proporcion_Fondos_Privados+Proporcion_Fondos_MAE+NGO_Country_Budget_Previous_Year+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_Africa+Vision_ONGD_Universal+Internacional+�+Delegacion,data=allInfoLog)
summary(model3)







#allInfoLog['Total_subvencion_en_el_Pais_y_Anyo'] <- log2(allInfoLog['Total_subvencion_en_el_Pais_y_Anyo']) 
#allInfoLog['Total_Fondos'] <- log2(allInfoLog['Total_Fondos'])
#allInfoLog['Subvencion_publica'] <- log2(allInfoLog['Subvencion_publica']) 
#allInfoLog['Gross_National_Income'] <- log2(allInfoLog['Gross_National_Income']) 



############################
modelbin1 <- glm(Visitado~ONU+Gross_National_Income+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Anyo_ONG+Internacional,data=allInfo,family = "binomial")
summary(modelbin1)

nullmod <- glm(Visitado~1,data=allInfo, family="binomial")
1-logLik(modelbin1)/logLik(nullmod)

modelbin1 <- glm(Visitado~ONU+Gross_National_Income+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_Africa+Vision_ONGD_Universal+Anyo_ONG+Internacional,data=allInfo,family = "binomial")
summary(modelbin1)

modelbin1 <- glm(Visitado~ONU+Gross_National_Income+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_Africa+Vision_ONGD_Universal+Anyo_ONG,data=allInfo,family = "binomial")
summary(modelbin1)


modelbin2 <- glm(Visitado~ONU+Gross_National_Income+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Anyo_ONG+Internacional,data=allInfoLog,family = "binomial")
summary(modelbin2)


modelbin2 <- glm(Visitado~ONU+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_Africa+Anyo_ONG+Internacional,data=allInfoLog,family = "binomial")
summary(modelbin2)

nullmod <- glm(Visitado~1,data=allInfo, family="binomial")
1-logLik(modelbin1)/logLik(nullmod)

nullmod2 <- glm(Visitado~1,data=allInfo, family="binomial")
1-logLik(modelbin2)/logLik(nullmod2)



res <- cor.test(allInfo$Subvencion_publica,allInfo$Fondos_Publicos_MAE, method = "pearson")
res


plot(log(allInfoPositius$Dinero_en_el_proyecto),log(allInfoPositius$Public_Grant))
plot(log(allInfoPositius$Dinero_en_el_proyecto),allInfoPositius$Total_subvencion_en_el_Pais_y_Anyo)

plot(log(allInfoPositius$Dinero_en_el_proyecto),log(allInfoPositius$Total_subvencion_en_el_Pais_y_Anyo))


d <- density(allInfoPositius$Gross_National_Income)
plot(d)
d <- density(log10(allInfoPositius$Gross_National_Income))
plot(d)
d <- density(allInfoPositius$Total_subvencion_en_el_Pais_y_Anyo)
plot(d)
d <- density(log(allInfoPositius$Total_subvencion_en_el_Pais_y_Anyo))
plot(d)
d <- density(allInfoPositius$Public_Grant)
plot(d)
d <- density(log(allInfoPositius$Public_Grant))
plot(d)

d <- density(allInfoPositius$Total_Fondos)
plot(d)

d <- density(log(allInfoPositius$Total_Fondos))
plot(d)


plot(log(allInfo$Fondos_Publicos_MAE),log(allInfo$Subvencion_publica))


plot(allInfo$Fondos_Publicos_MAE,allInfo$Dinero_en_el_proyecto)

plot(log(allInfo$Fondos_Publicos_MAE),log(allInfo$Dinero_en_el_proyecto))

plot(log(allInfo$dinero_anyo_anterior_en_proyectos),log(allInfo$Dinero_en_el_proyecto))