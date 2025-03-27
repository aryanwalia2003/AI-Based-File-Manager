import json
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()


# Azure LLM Initialization
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    azure_deployment="gpt-4o",
    api_version="2023-09-01-preview",
    api_key=os.getenv("AZURE_API_KEY"),
    temperature=0
)

def standardize_filename_with_llm(filename):
    """
    Calls the LLM to standardize a filename into camelCase format.
    
    - Removes underscores, dashes, and spaces.
    - Keeps the first word lowercase.
    - Capitalizes subsequent words.
    - Does not modify file extensions.
    """
    prompt = f"""
    Convert the following filename into camelCase format:
    - Remove spaces, underscores, and dashes.
    - Keep the first word lowercase.
    - Capitalize the first letter of each subsequent word.
    - Do not modify the file extension.
    - Do not give anything other than the camelCase name.

    Example:
    - "random_file.txt" → "randomFile.txt"
    - "My-Sample FILE.pdf" → "mySampleFile.pdf"
    - "user-data.json" → "userData.json"

    Filename: "{filename}"
    """

    # Extract LLM response
    response = llm([HumanMessage(content=prompt)]).content
    print(response)

    # try:
    #     response = llm([HumanMessage(content=prompt)]).content
    #     print("Raw LLM response:", response)
        
    #     # Clean up the response to ensure it's valid JSON
    #     # Remove any leading/trailing whitespace, comments or markdown formatting
    #     clean_response = response.strip()
        
    #     # If response is wrapped in markdown code blocks, extract just the JSON
    #     if clean_response.startswith("```json"):
    #         clean_response = clean_response.replace("```json", "", 1)
    #         if clean_response.endswith("```"):
    #             clean_response = clean_response[:-3]
    #         clean_response = clean_response.strip()
    #     elif clean_response.startswith("```"):
    #         clean_response = clean_response.replace("```", "", 1)
    #         if clean_response.endswith("```"):
    #             clean_response = clean_response[:-3]
    #         clean_response = clean_response.strip()
        
    #     # Try to parse the cleaned JSON
    #     parsed_response = json.loads(clean_response)
    #     return parsed_response
    # except json.JSONDecodeError as e:
    #     print(f"JSON decode error: {str(e)}")
    #     return {"error": f"Invalid JSON response from LLM: {str(e)}"}
    # except Exception as e:
    #     print(f"Unexpected error: {str(e)}")
    #     return {"error": f"Error processing LLM response: {str(e)}"}
    
    return response

def interpret_command(command: str):
    """
    Sends the user's command to the LLM and returns a structured JSON response.
    """
    # Note the double curly braces to escape them in the f-string
    prompt = f"""
    Interpret this command and return a structured JSON output. The commands will be of type , move , rename or standardise.
    Just return the JSON object, no extra text. Plus if any of the required feilds is missing write unspecified for that.
    If anyone gives vague file format or images make sure to append all the possible valid file formats into the filetype list.
    Or for eg if someone enters text files in file type add all the possible text file types like .docx,.txt,.pdf etc.

    Example format for action type = move:

    {{{{
    "action": "move",
    "file_type": "unspecified",
    "source": "C:/Users/Aryan Walia/OneDrive/Documents",
    "destination": "C:/Users/Aryan Walia/OneDrive/Desktop/Resumes",
    "keyword": "resume",
    "size_condition": "unspecified",
    "date_condition": "unspecified"
    }}}}

    Example format for action type= rename:
    
    {{{{
    "action": "rename",
    "directory": "C:/Users/Aryan/Documents",
    "lowercase": true,
    "replace_spaces": false,
    "remove_special_chars": false,
    "append_text": null,
    "prepend_text": null,
    "remove_text": null,
    }}}}
    
    Example Format for action type = standardise:
    
    {{{{
        "action": "standardise",
    }}}} 
    
    Example format for action type = categorise :
    {{{{
        "action": "categorise",
    }}}}

    Command: {command}
    """

    response = llm([HumanMessage(content=prompt)]).content
    print(response)

    try:
        response = llm([HumanMessage(content=prompt)]).content
        print("Raw LLM response:", response)
        
        # Clean up the response to ensure it's valid JSON
        # Remove any leading/trailing whitespace, comments or markdown formatting
        clean_response = response.strip()
        
        # If response is wrapped in markdown code blocks, extract just the JSON
        if clean_response.startswith("```json"):
            clean_response = clean_response.replace("```json", "", 1)
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
        elif clean_response.startswith("```"):
            clean_response = clean_response.replace("```", "", 1)
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
        
        # Try to parse the cleaned JSON
        parsed_response = json.loads(clean_response)
        return parsed_response
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return {"error": f"Invalid JSON response from LLM: {str(e)}"}
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return {"error": f"Error processing LLM response: {str(e)}"}