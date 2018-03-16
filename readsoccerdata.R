install.packages("RSQLite")
library(RSQLite)
#get connection
#download the db file from kaggle: https://www.kaggle.com/hugomathien/soccer/data
#mention the link where db has been downloaded in function below
  con <- dbConnect(drv=RSQLite::SQLite(), dbname="C:/Users/jaide/Downloads/soccer/database.sqlite")
  tables <- dbListTables(con)
#table names
  tables

#example of how to read from table
  #player_attr<-dbGetQuery(con,'select * from Player_Attributes')
  #player<-dbGetQuery(conn=con, statement=paste("SELECT * FROM '", tables[[4]], "'", sep=""))

#Read each table using the tables variable and create a dataframe with the 
#variable name as the table frame and write a csv file of the same name 
#in the location specified in write.csv function
for(i in tables){
  assign(i,dbGetQuery(conn=con, statement=paste("SELECT * FROM '", i, "'", sep="")))
  write.csv(get(i),file=paste("C:/Users/jaide/Downloads/soccer/",i,".csv",sep=""))
  #debugging
  #print(ls()[grep(pattern=paste("^",i,"$",sep=""), ls())])
  #print(grep(pattern=paste("^",i,"$",sep=""), ls()))
  }
