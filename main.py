import os
import shutil
import json
import re
from datetime import datetime
from llmHandler import interpret_command, standardize_filename_with_llm
from pdfCategorizer import process_pdfs,generate_class_label,generate_embeddings

LOG_FILE = "file_operations.log"

def log_action(action_details):
    """Logs the executed action into a log file."""
    with open(LOG_FILE, "a") as log:
        log.write(json.dumps(action_details) + "\n")

def parse_size_condition(size_condition):
    """Parses the size condition and returns (comparison_op, size_threshold_in_bytes)."""
    if size_condition in ["unspecified", "None", None]:
        return None, None

    size_units = {"MB": 1024 * 1024, "KB": 1024, "GB": 1024 * 1024 * 1024}
    
    for unit, multiplier in size_units.items():
        if size_condition.startswith((">=", ">", "<=", "<", "=")) and size_condition.endswith(unit):
            operator = size_condition[:-len(unit)].strip()
            size_value = int(operator[1:].strip()) * multiplier
            return operator[0], size_value

    print("Invalid size condition format. Use '>XMB' or '>=XMB'.")
    return None, None

def parse_date_condition(date_condition):
    """Parses the date condition and returns (comparison_op, datetime object)."""
    if date_condition in ["unspecified", "None", None]:
        return None, None

    try:
        operator = date_condition[0]
        date_str = date_condition[1:].strip()
        date_obj = datetime.strptime(date_str, "%B %Y")  # Format: "March 2022"
        return operator, date_obj
    except ValueError:
        print("Invalid date condition format. Use '>March 2022'.")
        return None, None

def get_file_creation_date(file_path):
    """Gets the creation/modification date of a file."""
    return datetime.fromtimestamp(os.path.getctime(file_path))  # Creation date

def filter_files_by_keyword(source, keyword):
    """
    Returns a list of files in 'source' that contain 'keyword' in their filename.
    If keyword is 'unspecified', returns all files.
    """
    if not os.path.exists(source):
        print(f"Error: Source folder '{source}' does not exist.")
        return []

    files = os.listdir(source)

    # If keyword is unspecified, return all files
    if keyword == "unspecified":
        return files

    keyword = keyword.lower()
    filtered_files = [file for file in files if keyword in file.lower()]
    
    return filtered_files

def rename_files(directory, lowercase=False, replace_spaces=False, remove_special_chars=False,
                 append_text=None, prepend_text=None,remove_text=None):
    """
    Renames files in a directory based on specified conditions.
    Uses a two-step approach to handle case-sensitivity issues on Windows.
    """
    import random

    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    files = os.listdir(directory)
    rename_log = []

    for file in files:
        old_path = os.path.join(directory, file)
        if not os.path.isfile(old_path):
            continue

        file_name, file_ext = os.path.splitext(file)
        new_name = file_name

        # Apply transformations
        if lowercase:
            new_name = new_name.lower()
        if replace_spaces:
            new_name = new_name.replace(" ", "")
        if remove_special_chars:
            new_name = re.sub(r'[^\w.-]', '', new_name)
        if prepend_text:
            new_name = prepend_text + new_name
        if append_text:
            new_name = new_name + append_text
        if remove_text:
            new_name = new_name.replace(remove_text, "")

        final_file = new_name + file_ext
        
        # If the name isn't changing, skip this file
        if file == final_file:
            continue

        # Two-step rename to handle Windows case-sensitivity issues
        # First rename to a temporary name with a random suffix
        temp_file = f"{file_name}_temp_{random.randint(10000, 99999)}{file_ext}"
        temp_path = os.path.join(directory, temp_file)
        
        # Then rename to final name
        final_path = os.path.join(directory, final_file)
        
        try:
            os.rename(old_path, temp_path)  # Step 1: Rename to temp name
            os.rename(temp_path, final_path)  # Step 2: Rename to final name
            rename_log.append({"old_name": file, "new_name": final_file})
            print(f"Renamed: {file} → {final_file}")
        except Exception as e:
            print(f"Error renaming {file}: {str(e)}")

    # Log the renaming action
    log_action({"action": "rename", "details": rename_log})

def standardize_filenames(directory):
    """
    Standardizes filenames in a directory to camelCase using LLM.
    """
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    files = os.listdir(directory)
    standardize_log = []

    for file in files:
        old_path = os.path.join(directory, file)
        if not os.path.isfile(old_path):
            continue

        # Get standardized filename from LLM
        standardized_name = standardize_filename_with_llm(file)
        
        # Skip if the name didn't change or if there was an error
        if file == standardized_name or isinstance(standardized_name, dict) and "error" in standardized_name:
            if isinstance(standardized_name, dict) and "error" in standardized_name:
                print(f"Error standardizing {file}: {standardized_name['error']}")
            continue
            
        # Use two-step renaming to handle case-sensitivity
        import random
        temp_file = f"temp_{random.randint(10000, 99999)}_{file}"
        temp_path = os.path.join(directory, temp_file)
        
        final_path = os.path.join(directory, standardized_name)
        
        try:
            os.rename(old_path, temp_path)  # Step 1: Rename to temp name
            os.rename(temp_path, final_path)  # Step 2: Rename to final name
            standardize_log.append({"old_name": file, "new_name": standardized_name})
            print(f"Standardized: {file} → {standardized_name}")
        except Exception as e:
            print(f"Error renaming {file}: {str(e)}")

    # Log the standardization action
    log_action({"action": "standardize", "details": standardize_log})

def execute_action(action_data):
    """Executes the file operation based on the parsed command."""
    action = action_data["action"]
    
    if action == "move":
        execute_move_action(action_data)
    elif action == "rename":
        execute_rename_action(action_data)
    elif action == "standardize" or action == "standardise":
        execute_standardize_action(action_data)
    else:
        print(f"Error: Unsupported action '{action}'!")

def execute_standardize_action(action_data):
    """Executes the file standardization operation."""
    directory = action_data.get("directory")
    
    if not directory:
        directory = input("Please enter the directory path to standardize filenames: ").strip()
    
    # Call the standardize_filenames function
    standardize_filenames(directory)
    
    print("File standardization operation completed.")

def execute_move_action(action_data):
    """Executes the file move operation."""
    file_type = action_data.get("file_type", "unspecified")
    source = action_data["source"]
    destination = action_data["destination"]
    size_condition = action_data.get("size_condition", "unspecified")
    date_condition = action_data.get("date_condition", "unspecified")
    keyword = action_data.get("keyword", "unspecified")

    # Ensure source exists
    if not os.path.exists(source):
        print(f"Error: Source folder '{source}' does not exist.")
        return

    # Ensure destination exists
    os.makedirs(destination, exist_ok=True)
    
    # Parse size and date conditions
    size_op, size_threshold = parse_size_condition(size_condition)
    date_op, date_threshold = parse_date_condition(date_condition)

    # Get files that match the keyword
    files_to_move = filter_files_by_keyword(source, keyword)

    # Convert file_type to list if it's a single string
    if isinstance(file_type, str) and file_type != "unspecified":
        file_type = [ext.strip() for ext in file_type.split(",")]

    filtered_files = []
    for file in files_to_move:
        file_path = os.path.join(source, file)

        # Skip directories
        if not os.path.isfile(file_path):
            continue

        # Check file type
        if file_type != "unspecified" and not any(file.endswith(ext) for ext in file_type):
            continue

        file_size = os.path.getsize(file_path)
        file_creation_date = get_file_creation_date(file_path)

        # Apply size condition
        if size_threshold is not None:
            if (size_op == ">" and file_size <= size_threshold) or \
               (size_op == ">=" and file_size < size_threshold) or \
               (size_op == "<" and file_size >= size_threshold) or \
               (size_op == "<=" and file_size > size_threshold) or \
               (size_op == "=" and file_size != size_threshold):
                continue
        
        # Apply date condition
        if date_threshold is not None:
            if (date_op == ">" and file_creation_date <= date_threshold) or \
               (date_op == ">=" and file_creation_date < date_threshold) or \
               (date_op == "<" and file_creation_date >= date_threshold) or \
               (date_op == "<=" and file_creation_date > date_threshold) or \
               (date_op == "=" and file_creation_date != date_threshold):
                continue

        filtered_files.append(file)

    if not filtered_files:
        print(f"No matching files found in {source}.")
        return

    print(f"Found {len(filtered_files)} files to move.")

    # Move each file
    for file in filtered_files:
        src_path = os.path.join(source, file)
        dest_path = os.path.join(destination, file)

        # Handle existing file conflict
        if os.path.exists(dest_path):
            user_choice = input(f"File '{file}' already exists in destination. Overwrite? (yes/no): ").strip().lower()
            if user_choice not in ["yes", "y"]:
                print(f"Skipping {file}.")
                continue

        shutil.move(src_path, dest_path)
        print(f"Moved: {file} -> {destination}")

    # Log the operation
    log_action(action_data)

    print("File move operation completed.")

def execute_rename_action(action_data):
    """Executes the file rename operation."""
    directory = action_data["directory"]
    lowercase = action_data["lowercase"]
    replace_spaces = action_data["replace_spaces"]
    remove_special_chars = action_data["remove_special_chars"]
    append_text = action_data["append_text"]
    prepend_text = action_data["prepend_text"]
    remove_text =action_data["remove_text"]

    # Call the rename_files function
    rename_files(
        directory=directory,
        lowercase=lowercase,
        replace_spaces=replace_spaces,
        remove_special_chars=remove_special_chars,
        append_text=append_text,
        prepend_text=prepend_text,
        remove_text=remove_text
    )

    print("File rename operation completed.")

def main():
    command = input("Enter a file management command: ").strip()
    print(f"Received command: {command}")

    # Get structured LLM response
    llm_response = interpret_command(command)

    # Handle potential LLM errors
    if "error" in llm_response:
        print("LLM Error:", llm_response["error"])
        return

    # Set default values for unspecified fields
    action = llm_response.get("action", "").lower()
    
    # Different field handling depending on action type
    if action == "move":
        for key in ["file_type", "size_condition", "date_condition", "keyword"]:
            if llm_response.get(key) in ["unspecified", "None", None]:
                llm_response[key] = "unspecified"
                
    elif action == "rename":
        # Set default values for rename parameters if not present
        rename_fields = ["lowercase", "replace_spaces", "remove_special_chars"]
        for field in rename_fields:
            if llm_response[field]=="unspecified":
                llm_response[field] = False
        
    
    elif action in ["standardize", "standardise"]:
        # Standardize is the American spelling, standardise is the British spelling
        # Both are accepted by converting to "standardize" internally
        llm_response["action"] = "standardize"
        if "directory" not in llm_response or llm_response["directory"] in ["unspecified", "None", None]:
            directory = input("Please enter the directory path to standardize filenames: ").strip()
            llm_response["directory"] = directory
    
    # Print action details for user to confirm
    print("\nAction Details:")
    for key, value in llm_response.items():
        if key != "confirmation":
            print(f"{key}: {value}")
    
    # Ask user for confirmation
    confirmation_msg = llm_response.get("confirmation", "Are you sure you want to proceed?")
    user_response = input(f"\n{confirmation_msg} (yes/no): ").strip().lower()
    if user_response not in ["yes", "y"]:
        print("Operation cancelled.")
        return

    # Execute the action
    execute_action(llm_response)

if __name__ == "__main__":
    main()