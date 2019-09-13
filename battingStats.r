library("dplyr")
library(ggplot2)
data(Batting, package="Lahman")

batting <- Batting
mlb <- filter(batting, lgID=="AL"|lgID=="NL")

mlb$battingAvg <- mlb$H/mlb$AB

head(mlb)

mlb <- filter(mlb, AB>=150 & yearID>=1903) #Completely abitrary number, but wanted to remove the .000 and the 1.000 BAs
                                           #1903 was the year the AL came into existence
summary(mlb)

#plot <- ggplot(mlb, aes(yearID, battingAvg, color = lgID))
#plot <- plot + geom_point(size = 5, alpha = .65)
#plot

yearStats <- data.frame(unique(mlb$yearID))
names(yearStats)[names(yearStats) == "unique.mlb.yearID."] <- "year"
yearStats$battingAvg = 0.0
yearStats$ALbattingAvg = 0.0
yearStats$NLbattingAvg = 0.0

for (row in 1:nrow(yearStats)){
  yearStats[row, "battingAvg"] <- mean(filter(mlb, yearID==yearStats[row, "year"])$battingAvg)
  yearStats[row, "ALbattingAvg"] <- mean(filter(mlb, yearID==yearStats[row, "year"] & lgID=="AL")$battingAvg)
  yearStats[row, "NLbattingAvg"] <- mean(filter(mlb, yearID==yearStats[row, "year"] & lgID=="NL")$battingAvg)
}

head(yearStats)
summary(yearStats)


p1 <- ggplot() + geom_line(data = yearStats, aes(y = battingAvg, x = year, color = "red")) 
p1 <- p1 + geom_line(data = yearStats, aes(x = year, y = ALbattingAvg, color = "green")) 
p1 <- p1 + geom_line(data = yearStats, aes(x = year, y = NLbattingAvg, color = "blue"))
p1 <- p1 + theme(axis.title = element_text(size = 15, color = "black", face = "bold")) + theme(plot.title = element_text(size = 30, face = "bold", vjust = 1)) + theme(axis.text = element_text(size = 13, face = "bold", color = "black")) + theme(legend.title = element_text(size = 12)) + theme(legend.text = element_text(size = 12))
p1 <- p1 + scale_color_discrete(name = "Leagues", labels = c("Overall", "American League", "National League"))

p1