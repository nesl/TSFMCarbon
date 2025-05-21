#!/bin/bash

# region_list=("BPAT" "FPL" "NYISO" "PL" "ES" "NL" "AUS_QLD")

region_list_1=("CISO" "PJM" "ERCO" "ISNE" "BPAT" "FPL" "NYISO" "SE" "DE" "PL" "ES" "NL" "AUS_QLD")
region_list_2=("Argentina" "Chile" "India-North" "India-South" "New-Zealand" "Nigeria" "Taiwan" "Uruguay")

region_list=("${region_list_1[@]}" "${region_list_2[@]}")

# Loop over each region and run the script
for region in "${region_list[@]}"; do
  echo "Running for region: $region"

  bash /kang/carbon/open_tsfm_carbon/time_series_library/scripts/long_term_forecast/ETT_script/TimeXer_carbon.sh "$region"
done


