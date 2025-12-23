#!/bin/bash

# Configuration
ORIGINAL_SCRIPT="slurm_job.sh"
PROJECT_ROOT=$(pwd)

# Check if original script exists
if [ ! -f "$ORIGINAL_SCRIPT" ]; then
    echo "Error: $ORIGINAL_SCRIPT not found in current directory."
    exit 1
fi

echo "=========================================================="
echo "Starting Batch Submission with Automatic Grouping"
echo "Project Root: $PROJECT_ROOT"
echo "=========================================================="

# Initialize variables
group_counter=1
# We start with group 1
current_group_dir="experiment_group_01"
mkdir -p "$current_group_dir"
mkdir -p "$current_group_dir/log"

# Function to parse and submit
process_table() {
    local first_group=true
    
    while IFS='|' read -r _ nodes mpi_per_node omp_per_mpi total _; do
        
        # 1. Clean whitespace
        nodes=$(echo "$nodes" | xargs)
        mpi_per_node=$(echo "$mpi_per_node" | xargs)
        omp_per_mpi=$(echo "$omp_per_mpi" | xargs)

        # 2. Detect Separator Lines (triggers new folder)
        if [[ "$nodes" == *"---"* ]]; then
            # If this isn't the very first separator at the top
            if [ "$first_group" = false ]; then
                ((group_counter++))
                current_group_dir=$(printf "experiment_group_%02d" $group_counter)
                echo "----------------------------------------------------------"
                echo "Creating new group directory: $current_group_dir"
                mkdir -p "$current_group_dir"
                mkdir -p "$current_group_dir/log"
            fi
            first_group=false
            continue
        fi

        # 3. Skip headers and empty lines
        if [[ "$nodes" == "nodes" ]] || [[ -z "$nodes" ]]; then continue; fi
        
        # 4. Skip the specific empty row in your table (where total is 0 or empty)
        if [[ -z "$mpi_per_node" ]] || [[ "$total" -eq 0 ]]; then continue; fi

        # 5. Prepare the localized SLURM script
        # We need to replace ./build with the absolute path so it runs from inside the subfolder
        # We assume the original script calls: ./build/mpd-plasma-dynamics
        
        LOCAL_SCRIPT="$current_group_dir/job.sh"
        
        # We create a copy of the script where we replace relative paths with absolute ones
        sed "s|\./build|$PROJECT_ROOT/build|g" "$ORIGINAL_SCRIPT" > "$LOCAL_SCRIPT"
        
        # 6. Define Job Name
        JOB_NAME="G${group_counter}-N${nodes}-M${mpi_per_node}-O${omp_per_mpi}"

        # 7. Submit from inside the folder
        # We enter the folder to submit so slurm-%j.out files end up there naturally
        cd "$current_group_dir" || exit

        sbatch \
            --nodes="$nodes" \
            --ntasks-per-node="$mpi_per_node" \
            --cpus-per-task="$omp_per_mpi" \
            --job-name="$JOB_NAME" \
            --mem=10G \
            --time=10:00:00 \
            --output="log/run-${JOB_NAME}-%j.out" \
            --error="log/run-${JOB_NAME}-%j.err" \
            "job.sh" > /dev/null

        cd "$PROJECT_ROOT" || exit

        printf "Submitted: Group %02d | Nodes: %-2s | MPI: %-2s | OMP: %-2s\n" "$group_counter" "$nodes" "$mpi_per_node" "$omp_per_mpi"
        
    done
}

# Feed the table into the function
process_table << 'EOF'
| nodes | MPI per node | OpenMP per MPI | Total |
|-------+--------------+----------------+-------|
|     2 |            1 |              1 |     2 |
|     2 |            1 |              2 |     4 |
|     2 |            1 |              4 |     8 |
|     2 |            1 |              8 |    16 |
|     2 |            1 |             16 |    32 |
|     2 |            1 |             24 |    48 |
|     2 |            1 |             32 |    64 |
|     2 |            1 |             40 |    80 |
|     2 |            1 |             48 |    96 |
|-------+--------------+----------------+-------|
|     2 |            1 |              1 |     2 |
|     2 |            2 |              1 |     4 |
|     2 |            4 |              1 |     8 |
|     2 |            8 |              1 |    16 |
|     2 |           16 |              1 |    32 |
|     2 |           24 |              1 |    48 |
|     2 |           32 |              1 |    64 |
|     2 |           40 |              1 |    80 |
|     2 |           48 |              1 |    96 |
EOF

# |     1 |            1 |              1 |     3 |
# |     1 |            1 |              2 |     2 |
# |     1 |            1 |              4 |     4 |
# |     1 |            1 |              8 |     8 |
# |     1 |            1 |             16 |    16 |
# |     1 |            1 |             24 |    24 |
# |     1 |            1 |             32 |    32 |
# |     1 |            1 |             40 |    40 |
# |     1 |            1 |             48 |    48 |
# |-------+--------------+----------------+-------|
# |     1 |            1 |              1 |     1 |
# |     1 |            2 |              1 |     2 |
# |     1 |            4 |              1 |     4 |
# |     1 |            8 |              1 |     8 |
# |     1 |           16 |              1 |    16 |
# |     1 |           24 |              1 |    24 |
# |     1 |           32 |              1 |    32 |
# |     1 |           40 |              1 |    40 |
# |     1 |           48 |              1 |    48 |
# |       |              |                |     0 |
# |     2 |            1 |              1 |     2 |
# |     2 |            2 |              1 |     4 |
# |     2 |            4 |              1 |     8 |
# |     2 |            8 |              1 |    16 |
# |     2 |           16 |              1 |    32 |
# |     2 |           24 |              1 |    48 |
# |     2 |           32 |              1 |    64 |
# |     2 |           40 |              1 |    80 |
# |     2 |           48 |              1 |    96 |
# |-------+--------------+----------------+-------|
# |     1 |            1 |              2 |     2 |
# |     1 |            2 |              2 |     4 |
# |     1 |            4 |              2 |     8 |
# |     1 |            8 |              2 |    16 |
# |     1 |           16 |              2 |    32 |
# |     1 |           24 |              2 |    48 |
# |     2 |           16 |              2 |    64 |
# |     2 |           20 |              2 |    80 |
# |     2 |           24 |              2 |    96 |
# |-------+--------------+----------------+-------|
# |     1 |            1 |              4 |     4 |
# |     1 |            2 |              4 |     8 |
# |     1 |            4 |              4 |    16 |
# |     1 |            8 |              4 |    32 |
# |     1 |           12 |              4 |    48 |
# |     2 |            8 |              4 |    64 |
# |     2 |           10 |              4 |    80 |
# |     2 |           12 |              4 |    96 |
# |-------+--------------+----------------+-------|
# |     1 |            1 |              8 |     8 |
# |     1 |            2 |              8 |    16 |
# |     1 |            3 |              8 |    24 |
# |     1 |            4 |              8 |    32 |
# |     1 |            6 |              8 |    48 |
# |     2 |            4 |              8 |    64 |
# |     2 |            5 |              8 |    80 |
# |     2 |            6 |              8 |    96 |
# |-------+--------------+----------------+-------|
# |     1 |            1 |             16 |    16 |
# |     1 |            2 |             16 |    32 |
# |     1 |            3 |             16 |    48 |
# |     2 |            2 |             16 |    64 |
# |     2 |            3 |             16 |    96 |

echo "=========================================================="
echo "All jobs submitted."
