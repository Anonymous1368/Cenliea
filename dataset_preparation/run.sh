#!/bin/sh
# Record the start time for the entire batch
batch_start_time=$(date +%s)
echo "Running all scripts..."

# Get the current date and time as the batch start time in the format YYYYMMDD_HHMMSS
batch_start_time_str=$(date +"%Y%m%d_%H%M%S")


#To run 3_merge_and_shard_datasets.py script:
# log_file_dir=$(python -c "import Param; print(Param.MIXED_DATASET_PATH)")
# To run scripts 1,2, 3:
# log_file_dir=$(python -c "import Param; print(Param.INPUT_PATH)")

# Construct the full log file path with timestamped name
# log_file="${log_file_dir}/log_${batch_start_time_str}.log"
# To run scripts 1,2, 3:
# log_file="${log_file_dir}log_${batch_start_time_str}.log" #Having no slash



#python -u 1_convert_rdf_to_ntriples.py
#python -u 2_generate_structured_pos_neg_samples.py.py
#python -u 3_merge_and_shard_datasets.py

# List of Python files to execute
scripts=("1_convert_rdf_to_ntriples.py" "2_generate_structured_pos_neg_samples.py" "3_merge_and_shard_datasets.py")  # or single run: ("1_convert_rdf_to_ntriples.py") 


# Run each script and measure individual runtimes
for script in "${scripts[@]}"; do

    # Load the log file directory from params.py
    # Conditional logic to determine which path to load
    if [[ "$script" == "3_merge_and_shard_datasets.py" ]]; then
        # For 3_integrate_train_test_jsons.py, use MIXED_DATASET_PATH
        log_file_dir=$(python -c "import Param; print(Param.MIXED_DATASET_PATH)")
        log_file="${log_file_dir}/log_${batch_start_time_str}.log"
    else
        # For other scripts, use INPUT_PATH
        log_file_dir=$(python -c "import Param; print(Param.INPUT_PATH)")
        log_file="${log_file_dir}log_${batch_start_time_str}.log" #HAving no slash
    fi

    # Create the directory if it doesn't exist
    mkdir -p "$log_file_dir"

    # Start logging
    echo "Starting script execution..." > "$log_file"
    echo "------------------------------------------------------" >> "$log_file"
    echo "Running $script..." | tee -a "$log_file"

    # Record start time for each individual script
    start_time=$(date +%s)

    # Run the script and capture its output to the log file
    python "$script" &>> "$log_file"

    # Calculate and log individual script runtime
    end_time=$(date +%s)
    runtime=$((end_time - start_time))
    echo "Runtime for $script: $runtime seconds" | tee -a "$log_file"
    echo "------------------------------------------------------" | tee -a "$log_file"
done

# Record the end time for the entire batch
batch_end_time=$(date +%s)

# Calculate and log the total runtime for all scripts
total_runtime=$((batch_end_time - batch_start_time))
echo "Total runtime for all scripts: $total_runtime seconds" | tee -a "$log_file"
echo "Execution complete. Outputs saved to $log_file."
