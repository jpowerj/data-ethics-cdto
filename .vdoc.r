#
library(tidyverse)
bios <- paste0(readLines("bios.txt"), collapse="\n")
str_split(bios, "---")
# length(bios)
# for (bio_index in seq_along(bios)) {
#   cur_bio <- bios[[bio_index]]
#   print(bio_index)
#   writeLines(cur_bio)
# }
#portfolio_df |> select("Bio") |> head(2)
#
#
#
#
#
#
#
#
#
#
