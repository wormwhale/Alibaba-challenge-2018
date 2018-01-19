library(data.table)
library(stringr)
library(stringi)
library(plyr)


for(i in list.files('./data')){
  if(!str_detect(i, '^C')){
   df <- fread(paste('data/', i, sep=''))
   unique(df$date_id) %>>% sapply(function(x){
     fwrite(df[date_id==x], paste('data/partition/', str_replace(i, '.csv', ''), '_', x, '.csv', sep=''))  
   })
   
  }
}

