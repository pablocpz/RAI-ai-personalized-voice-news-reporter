import os

# Set the directory containing the .md files
directory = 'rag_docs'

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the file has a .md extension
    if filename.endswith('.md'):
        # Define the new filename (shortened)
        new_filename = filename[:10] + '.md'  # Example: shorten to the first 10 characters
        
        # Get the full path for both old and new filenames
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {filename} -> {new_filename}')
