setwd("C:/Users/bcoma/Documents/GitHub/Tesis_UB/scripts")
library(readxl)


allInfo <- read_excel('../output/allExcels_negatiu.xlsx')
allInfoPositius <- allInfo[allInfo$Visitado==1,]

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



allInfo["Anyo_ONG"]= 2020-allInfo["Anyo_ONG"]
model1 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Anyo_ONG+Internacional,data=allInfo)
summary(model1)

model1 <- lm(Dinero_en_el_proyecto~Gross_National_Income+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year,data=allInfo)
summary(model1)


allInfoLog <- allInfo[,]
allInfoLog['Dinero_en_el_proyecto'] <- log2(allInfoLog['Dinero_en_el_proyecto']) 
allInfoLog['NGO_Country_Budget_Previous_Year'] <- log2(allInfoLog['NGO_Country_Budget_Previous_Year'])
allInfoLog['Total_subvencion_en_el_Pais_y_Anyo'] <- log2(allInfoLog['Total_subvencion_en_el_Pais_y_Anyo'])
allInfoLog['Gross_National_Income'] <- log2(allInfoLog['Gross_National_Income'])
allInfoLog['Public_Grant'] <- log2(allInfoLog['Public_Grant'])

allInfoLog[allInfoLog<0] <- 0

model2 <- lm(Dinero_en_el_proyecto~ONU+Gross_National_Income+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Anyo_ONG+Internacional,data=allInfoLog)
summary(model2)

model2 <- lm(Dinero_en_el_proyecto~Total_Fondos+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Vision_ONGD_Africa+Anyo_ONG+Internacional,data=allInfoLog)
summary(model2)



#allInfoLog['Total_subvencion_en_el_Pais_y_Anyo'] <- log2(allInfoLog['Total_subvencion_en_el_Pais_y_Anyo']) 
#allInfoLog['Total_Fondos'] <- log2(allInfoLog['Total_Fondos'])
#allInfoLog['Subvencion_publica'] <- log2(allInfoLog['Subvencion_publica']) 
#allInfoLog['Gross_National_Income'] <- log2(allInfoLog['Gross_National_Income']) 



############################
modelbin1 <- glm(Visitado~ONU+Gross_National_Income+Total_Fondos+Public_Grant+NGO_Country_Budget_Previous_Year+Proporcion_Fondos_Privados+Total_subvencion_en_el_Pais_y_Anyo+Vision_ONGD_LatinAmerica+Vision_ONGD_Africa+Vision_ONGD_Universal+Anyo_ONG+Internacional,data=allInfo,family = "binomial")
summary(modelbin1)

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