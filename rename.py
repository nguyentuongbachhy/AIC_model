import os

# Set the directory path
directory = 'D:/AIC/model/assets/results'

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Construct the full file path
    old_file = os.path.join(directory, filename)

    new_filename = old_file.replace('n_details.json', 'n')

    new_file = os.path.join(directory, new_filename)
    
    # Rename the file
    os.rename(old_file, new_file)

print("Files renamed successfully!")