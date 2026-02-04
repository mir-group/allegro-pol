# ==============
# COMBINE FILES
# ==============

"""
Author: S. Falletta

This script combines LAMMPS file obtained with multiple re-runs,
which share the same root. The format of files is: {root}_{i}
where {i} is an integer number representing the starting step.
All these files must be located in the same folder.
"""

import os
import re

# Define the root of the files to search for
root_name = "log.hyst_BaTiO3-E0_BaTiO3_supercell_init_300_1_4_1000000_"

# Initialize the output file
infile_tot = "BaTiO3.dat"
open(infile_tot, "w").close()  # This clears the file if it exists

# Get all files with the given root name in the current folder
all_files = [f for f in os.listdir(".") if f.startswith(root_name)]

# Extract the step values from the filenames
step_values = sorted([int(re.search(r"_(\d+)$", f).group(1)) for f in all_files])

# List of input files sorted by their step values
files = [f"{root_name}{step}" for step in step_values]

# Define tau as the difference between consecutive step values
tau_values = [step_values[i] - step_values[i - 1] for i in range(1, len(step_values))]
tau_values.insert(0, step_values[0])  # Add the first tau (from 0 to the first step)

# Loop through input files
for i, infile in enumerate(files):
    print(f"Processing file: {infile}")

    start_step = step_values[i]  # Starting point for the current infile

    # Handle the last file differently
    if i == len(files) - 1:
        end_step = float("inf")  # Set a large number to capture all remaining steps
    else:
        end_step = step_values[i + 1] - 1  # Ending point

    # Step 1: Extract lines after the header
    # Find the line number containing the header
    cmd_grep = f"grep -n 'Step.*Time.*Temp.*Press.*PotEng' {infile} | cut -d: -f1"
    header_line = int(os.popen(cmd_grep).read().strip())

    # Use sed to extract lines starting from the header line
    cmd_sed = f"sed -n '{header_line},$p' {infile} > temp.dat"
    os.system(cmd_sed)

    if i == 0:
        # Append lines above the header (without the header line itself)
        cmd_sed_above = f"sed -n '1,{header_line}p' {infile} > {infile_tot}"
        os.system(cmd_sed_above)

        # Append the lines below the header, but only up to the first tau
        cmd_below = f"sed -n '{header_line + 1},$p' {infile} | awk '$1 <= {end_step}' >> {infile_tot}"
        os.system(cmd_below)

    elif i < len(files) - 1:
        # For subsequent infiles, keep lines with steps between tau increments
        cmd_awk = (
            f"awk '$1 >= {start_step} && $1 <= {end_step}' temp.dat >> {infile_tot}"
        )
        os.system(cmd_awk)

    else:
        # For the last file, append all lines starting from start_step, but exclude "Loop time" and below
        # cmd_awk_last = f"awk '$1 >= {start_step} && !found {{if (/Loop time/) found=1; else print;}}' temp.dat >> {infile_tot}"
        cmd_awk_last = f"awk '$1 >= {start_step} && !found {{if (/Loop time/) found=1; else if (printed++) print;}}' temp.dat >> {infile_tot}"

        os.system(cmd_awk_last)

    # Clean up temp file
    os.remove("temp.dat")

print(f"All files processed and appended to {infile_tot}.")

# Check that there are no missing steps
header_line = int(
    os.popen(f"grep -n 'Step.*Time.*Temp.*Press.*PotEng' {infile_tot} | cut -d: -f1")
    .read()
    .strip()
)
step_lines = (
    os.popen(f"sed -n '{header_line + 1},$p' {infile_tot} | awk '{{print $1}}'")
    .read()
    .strip()
    .split("\n")
)

# Convert to list of integers
step_numbers = [int(step) for step in step_lines if step.strip()]

# Read the current file contents
with open(infile_tot, "r") as f:
    content = f.readlines()

# Ensure there are no empty lines after the header
# Remove any empty lines starting from header_line + 1
content = [
    line for idx, line in enumerate(content) if idx <= header_line or line.strip()
]

# Check for missing steps and add them if necessary
spacing_check = True
if len(step_numbers) > 1:
    reference_diff = (
        step_numbers[1] - step_numbers[0]
    )  # Difference between first and second steps
    time_step_diff = float(content[header_line + 1].split()[1]) - float(
        content[header_line].split()[1]
    )  # Calculate time step increment

    # Initialize an offset to account for changes in content length
    offset = 0
    iter = 0

    for i in range(1, len(step_numbers) - 1):
        # Check for spacing inconsistencies
        if step_numbers[i + 1] - step_numbers[i] != reference_diff:
            spacing_check = False
            print(
                f"\nMismatch found between steps {step_numbers[i]} and {step_numbers[i + 1]}: difference is {step_numbers[i + 1] - step_numbers[i]} (expected {reference_diff})"
            )

            # Adjust index for the current line based on previous insertions
            current_line_idx = header_line + i + offset - iter

            # Identify the previous line (to copy) and delete the incorrect step line
            previous_line = content[current_line_idx - 1]  # Line for step_numbers[i-1]
            incorrect_step_line = content[current_line_idx]  # The incorrect line
            # print("incorrect line:", incorrect_step_line)
            content.pop(current_line_idx)  # Remove the incorrect step line

            # Get previous step and time values to increment
            previous_line_columns = previous_line.split()
            previous_step = int(previous_line_columns[0])
            previous_time = float(previous_line_columns[1])

            # Add missing steps using the modified previous line
            missing_steps = list(
                range(step_numbers[i], step_numbers[i + 1], reference_diff)
            )
            if missing_steps:
                print(
                    f"Adding {len(missing_steps)} missing steps between {missing_steps[0]} and {missing_steps[-1]}"
                )

                # Insert missing steps in the correct position by copying the previous line
                for step in missing_steps:
                    # Increment both step number and time
                    new_step = previous_step + reference_diff
                    new_time = previous_time + time_step_diff

                    # Replace step and time values in the copied line and reformat with proper columns
                    line_to_insert = (
                        f"    {new_step:<8} {new_time:<12.2f} "
                        + " ".join(f"{col:<12}" for col in previous_line_columns[2:])
                        + " # extra\n"
                    )

                    # Insert the new line at the current adjusted index
                    content.insert(current_line_idx, line_to_insert)

                    # Update previous step and time for the next missing step
                    previous_step = new_step
                    previous_time = new_time

                    # Adjust the current index and offset for the inserted line
                    current_line_idx += 1
                    offset += 1

                # Since the line was replaced, shift the index backward by 1 for the next iteration
                i -= 1
                iter += 1

# Write the updated content back to the file
with open(infile_tot, "w") as f:
    f.writelines(content)

if spacing_check:
    print("All consecutive MD steps differ by 10.")
