#!/bin/bash

# Define base directory and method name
base_dir="/kang/carbon/tsfm_carbon"
# method_name="autoarima"
method_name="autoarima"

# Define region list
# region_list=("CISO" "PJM" "ERCO" "ISNE" "SE" "DE")
# region_list=("BPAT" "FPL" "NYISO" "PL" "ES" "NL" "AUS_QLD")
# region_list=("Argentina" "Chile" "India-North" "Nigeria" "Taiwan")

region_list_1=("CISO" "PJM" "ERCO" "ISNE" "BPAT" "FPL" "NYISO" "SE" "DE" "PL" "ES" "NL" "AUS_QLD")
region_list_2=("Argentina" "Chile" "India-North" "India-South" "New-Zealand" "Nigeria" "Taiwan" "Uruguay")

region_list=("${region_list_1[@]}" "${region_list_2[@]}")

# region="PL"
# Log directory and file setup
logdir="${base_dir}/logs"
mkdir -p "$logdir"

timestamp=$(date +%Y%m%d_%H%M%S)
log_file="${logdir}/tsfms_${method_name}_${timestamp}.log"

# Define Python interpreter and script path dynamically based on method name
# python_path="${base_dir}/.${method_name}/bin/python"
python_path="/kang/carbon/TSFM_Building_carbon/.moment/bin/python"

# main_carbon_autoarima.py
file_path="${base_dir}/main_carbon_${method_name}.py"

echo "Logging to $log_file"

# Loop through each region
for region in "${region_list[@]}"; do
  echo "Running for region: $region"
  "$python_path" "$file_path" --region "$region" | tee -a "$log_file"
done


: '

# Logs are stored in: /kang/carbon/tsfm_carbon/logs/auroarima.log

mkdir -p /kang/carbon/tsfm_carbon/logs/


nohup bash /kang/carbon/tsfm_carbon/scripts/run_auroarima.sh > /kang/carbon/tsfm_carbon/logs/auroarima.log 2>&1 &

'



