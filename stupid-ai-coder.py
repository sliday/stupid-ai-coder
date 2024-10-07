import os
import asyncio
import logging
from typing import List, Optional, Tuple
from difflib import unified_diff
from tqdm import tqdm
import concurrent.futures
import ell
import traceback
from colorama import init, Fore, Back, Style
import emoji
import subprocess
import shutil
import difflib
import ast
import sys
import time
import threading
import signal
import psutil

# Initialize colorama for cross-platform colored terminal output
init(autoreset=True)

# Custom logging formatter for colorful console output
class ColorfulFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Back.RED + Fore.WHITE + "%(asctime)s - %(levelname)s - %(message)s" + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Initialize ell with versioning and autocommit
ell.init(store='./logdir', autocommit=True, verbose=False)

# Configure logging
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the Spinner class implementation:
class Spinner:
    def __init__(self):
        self.spinning = False
        self.spinner_chars = ['|', '/', '-', '\\']
        self.current = 0

    def spin(self):
        while self.spinning:
            sys.stdout.write(f"\r{self.spinner_chars[self.current]} Processing...")
            sys.stdout.flush()
            self.current = (self.current + 1) % len(self.spinner_chars)
            time.sleep(0.1)

    def __enter__(self):
        self.spinning = True
        threading.Thread(target=self.spin, daemon=True).start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.spinning = False
        sys.stdout.write('\r')
        sys.stdout.flush()

def track_progress(current: int, total: int):
    """Display a colorful progress bar for the current operation."""
    with tqdm(total=total, bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)) as pbar:
        pbar.update(current)

def get_model_choice():
    """Prompt the user to choose an AI model with a beautified interface."""
    models = [
        ("1", "claude-3-5-sonnet-20240620", "Default"),
        ("2", "gpt-4o-mini", "Alternative"),
        ("3", "llama3.2", "Local Ollama")
    ]
    
    print(emoji.emojize(f"\n{Fore.CYAN}:robot: Model Selection Menu {Fore.RESET}"))
    print(f"{Fore.YELLOW}{'=' * 40}{Fore.RESET}")
    
    for number, model, description in models:
        print(f"{Fore.GREEN}{number}. {Fore.BLUE}{model:<30}{Fore.YELLOW}[{description}]{Fore.RESET}")
    
    print(f"{Fore.YELLOW}{'=' * 40}{Fore.RESET}\n")

    while True:
        choice = input(emoji.emojize(f"{Fore.MAGENTA}:gear: Enter your choice (1-3) or press Enter for default: {Fore.RESET}"))
        
        if choice == '' or choice == '1':
            return "claude-3-5-sonnet-20240620"
        elif choice == '2':
            return "gpt-4o-mini"
        elif choice == '3':
            return "llama3.2"
        else:
            print(emoji.emojize(f"{Fore.RED}:warning: Invalid choice. Please enter 1, 2, 3, or press Enter for default.{Fore.RESET}"))

@ell.simple(model="gpt-4o-mini", max_tokens=16384)  # Default model, will be overridden in main()
def generate_initial_code(task: str, model: str) -> str:
    """Generate initial code for the given task."""
    prompt = f"""
    You are the world's best expert full-stack programmer, recognized as a Google L5 level PYTHON engineer. 
    Your task is to generate initial boilerplate code for the following task:

    <task>
    {task}
    </task>

    The code should include basic structure and placeholders for main functionality.
    Provide comprehensive comments explaining the code structure and functionality.
    Ensure the code is well-organized and follows best practices.
    """
    return prompt

@ell.simple(model="gpt-4o-mini", max_tokens=16384)  # Default model, will be overridden in main()
def improve_code_base(code: str, iteration: int, previous_versions: str, model: str, error: Optional[str] = None) -> str:
    """Improve the given Python code using AI assistance."""
    error_prompt = f"\n\nThe previous version of the code produced the following error:\n{error}\nPlease fix this error and improve the code." if error else ""
    
    prompt = f"""
    You are the world's best expert full-stack programmer, recognized as a Google L5 level software engineer. 
    Your task is to improve the given Python code, making it more efficient, readable, and adding 1 new feature.

    Current code:
    <code>
    {code}
    </code>

    Iteration: {iteration}

    CONTEXT (Previous versions):
    <context>
    {previous_versions}
    </context>

    Possble errors (ignore if no error):
    <error>
    {error_prompt}
    </error>

    Please improve the code:
    1. Prioritize creativity
    2. Optmize for readability and maintainability
    3. MUST add 1 new mega ultra cool feature that is not in the previous versions
    Provide the improved code wrapped in <answer> tags.

    METHODOLOGY:
    <methodology>
    Begin by enclosing all thoughts within <thinking> tags, exploring multiple angles and approaches.
    Break down the solution into clear steps within <step> tags. Start with a 20-step budget, requesting more for complex problems if needed.
    Use <count> tags after each step to show the remaining budget. Stop when reaching 0.
    Continuously adjust your reasoning based on intermediate results and reflections, adapting your strategy as you progress.
    Regularly evaluate progress using <reflection> tags. Be critical and honest about your reasoning process.
    Assign a quality score between 0.0 and 1.0 using <reward> tags after each reflection. Use this to guide your approach:

    0.8+: Continue current approach
    0.5-0.7: Consider minor adjustments
    Below 0.5: Seriously consider backtracking and trying a different approach

    If unsure or if reward score is low, backtrack and try a different approach, explaining your decision within <thinking> tags.
    For mathematical problems, show all work explicitly using LaTeX for formal notation and provide detailed proofs.
    Explore multiple solutions individually if possible, comparing approaches in reflections.
    Use thoughts as a scratchpad, writing out all calculations and reasoning explicitly.
    Synthesize the final answer within <answer> tags, providing a clear, concise summary.
    Conclude with a final reflection on the overall solution, discussing effectiveness, challenges, and solutions. Assign a final reward score.
    No "Here is the improved code", no `\`\`\`python` tags, no "```" anywhere.
    </methodology>
    """
    return prompt

@ell.simple(model="gpt-4o-mini", max_tokens=50)
def generate_title(task: str) -> str:
    """Generate a 2-word title for the given task."""
    return f"Generate a concise 2-word title for the following task: {task}. No comments, no markdown, no code, no ```python, no <answer>, simply the title"

@ell.simple(model="gpt-4o-mini", max_tokens=4192)
def generate_readme(task: str, title: str, filename: str) -> str:
    """Generate a README.md file for the project."""
    prompt = f"""
    Create a README.md file for a Python project with the following details:
    - Project Title: {title}
    - Task Description: {task}
    - Main File: {filename}

    The README should include:
    1. A brief description of the project
    2. How to run the code
    3. Any dependencies required
    4. A short explanation of what the code does
    5. Other notes and helpful information

    Provide the content of the README.md file, formatted in Markdown.
    No comments, no extra text, no \`\`\`markdown, simply the README.md content.
    """
    return prompt

async def get_previous_versions(current_iteration: int) -> str:
    """Retrieve diffs of previous code versions."""
    diffs = ""
    previous_content = ""
    try:
        logger.debug(f"Retrieving diffs up to iteration {current_iteration}")
        for i in range(1, current_iteration):
            filename = f"monolyth{i}.py"
            if os.path.exists(filename):
                async with asyncio.Lock():
                    with open(filename, "r") as f:
                        current_content = await asyncio.to_thread(f.read)
                        if previous_content:
                            diff = difflib.unified_diff(
                                previous_content.splitlines(keepends=True),
                                current_content.splitlines(keepends=True),
                                fromfile=f'monolyth{i-1}.py',
                                tofile=filename
                            )
                            diffs += f"===Diff v{i-1} to v{i}===\n{''.join(diff)}\n\n"
                        previous_content = current_content
        logger.debug(f"Retrieved {current_iteration - 2} diffs")
    except Exception as e:
        logger.error(f"Error in get_previous_versions: {str(e)}")
        logger.error(traceback.format_exc())
    return diffs

def visualize_diff(old_code: str, new_code: str) -> str:
    """Generate a unified diff between old and new code versions."""
    try:
        logger.debug("Generating diff between old and new code versions")
        diff = unified_diff(
            old_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile='Previous Version',
            tofile='New Version',
            n=3
        )
        return ''.join(diff)
    except Exception as e:
        logger.error(f"Error in visualize_diff: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error generating diff: {str(e)}"

async def improve_code_smart(code: str, iteration: int, previous_versions: str, model: str, error: Optional[str] = None, use_parallel: bool = False) -> str:
    """Improve code using AI, with option for parallel processing."""
    try:
        with Spinner():  # Add this line to create a spinner
            if use_parallel:
                logger.debug(f"Starting parallel code improvement for iteration {iteration}")
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    future = executor.submit(improve_code_base, code, iteration, previous_versions, model, error)
                    result = await asyncio.to_thread(future.result)
                logger.debug(f"Parallel code improvement completed for iteration {iteration}")
            else:
                logger.debug(f"Starting sequential code improvement for iteration {iteration}")
                result = improve_code_base(code, iteration, previous_versions, model, error)
                logger.debug(f"Sequential code improvement completed for iteration {iteration}")
        return result
    except Exception as e:
        logger.error(f"Error in improve_code_smart: {str(e)}")
        logger.error(traceback.format_exc())
        return f"<answer>Error in execution: {str(e)}</answer>"

def generate_requirements(folder_name: str):
    """
    Read the Python script in the project folder, analyze its imports,
    and generate a requirements.txt file in the same folder.
    """
    project_files = [f for f in os.listdir(folder_name) if f.endswith('.py')]
    if not project_files:
        logger.warning(f"No Python files found in {folder_name}")
        return

    imports = set()
    for file in project_files:
        file_path = os.path.join(folder_name, file)
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
    
    # Filter out standard library modules
    third_party_imports = [imp for imp in imports if imp not in sys.builtin_module_names]
    
    requirements_path = os.path.join(folder_name, 'requirements.txt')
    with open(requirements_path, 'w') as req_file:
        for module in sorted(third_party_imports):
            req_file.write(f"{module}\n")

    logger.info(emoji.emojize(f":memo: requirements.txt file has been generated in {folder_name}"))

@ell.simple(model="gpt-4o-mini")
def summarize_changes(diff: str) -> str:
    """You are an expert code reviewer. Summarize the key changes in the provided diff concisely."""
    return f"Summarize the following code changes:\n\n{diff}"

# Function to get diff statistics
def get_diff_stats(diff: str) -> dict:
    lines = diff.split('\n')
    additions = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))
    return {
        "total_changes": additions + deletions,
        "additions": additions,
        "deletions": deletions
    }

def run_with_timeout(cmd: list, timeout: int = 10) -> Tuple[Optional[str], Optional[str], int]:
    """Run a command with timeout and return stdout, stderr, and return code."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid
        )
        start_time = time.time()
        
        while process.poll() is None:
            if time.time() - start_time > timeout:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(0.1)  # Give it a moment to terminate gracefully
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                return None, "Execution timed out and was forcefully terminated", -1
            time.sleep(0.1)
        
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode
    except Exception as e:
        return None, f"Error running process: {str(e)}", -1

def kill_child_processes(parent_pid):
    """Terminate all child processes of a given parent process."""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, still_alive = psutil.wait_procs(children, timeout=3)
    for p in still_alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass

def test_solution(filename: str) -> Optional[str]:
    """Test the generated solution by running it and capturing any errors."""
    try:
        stdout, stderr, return_code = run_with_timeout(['python3', filename])
        
        # Write both stdout and stderr to output.txt
        with open('output.txt', 'w') as f:
            f.write("STDOUT:\n")
            f.write(stdout or "")
            f.write("\nSTDERR:\n")
            f.write(stderr or "")
        
        # Check for errors in both return code and output content
        if return_code != 0:
            return f"Execution failed with return code {return_code}. Error: {stderr}"
        
        # Check for common error patterns in stdout and stderr
        error_patterns = ['Exception', 'Error:', 'Traceback', 'ERROR', 'error', 'SyntaxError', 'NameError', 'TypeError']
        combined_output = (stdout or "") + (stderr or "")
        for pattern in error_patterns:
            if pattern in combined_output:
                return f"Potential error detected: '{pattern}'. Check output.txt for details."
        
        # Check for infinite loops or excessive resource usage
        if "Execution timed out" in stderr:
            return "Possible infinite loop or excessive runtime detected."
        
        return None
    except Exception as e:
        logger.error(f"Error in test_solution: {str(e)}")
        return f"Unexpected error during testing: {str(e)}"
    finally:
        # Ensure all child processes are terminated
        kill_child_processes(os.getpid())

async def main():
    """Main function to orchestrate the AI code generation process."""
    try:
        print(emoji.emojize(f"{Fore.CYAN}:robot: Welcome to the Monolyth AI Code Generator! :rocket:{Style.RESET_ALL}"))
        task = input(emoji.emojize(f"{Fore.YELLOW}:thought_balloon: What should I code? (Example: code tetris game) {Style.RESET_ALL}"))
        iterations = int(input(emoji.emojize(f"{Fore.YELLOW}🔁 How many iterations? (up to 20) {Style.RESET_ALL}")))
        
        if iterations < 1 or iterations > 20:
            raise ValueError("Iterations must be between 1 and 20.")

        model = get_model_choice()
        
        # Override the model for the LMPs
        generate_initial_code.model = model
        improve_code_base.model = model
        generate_title.model = model
        generate_readme.model = model
        
        print(f"\n{Fore.CYAN}Starting code generation process...{Style.RESET_ALL}")
        
        # Generate the 2-word title and create folder
        print(f"{Fore.YELLOW}Generating project title...{Style.RESET_ALL}")
        title = generate_title(task).strip()
        folder_name = title.replace(" ", "_")
        os.makedirs(folder_name, exist_ok=True)
        print(f"{Fore.GREEN}Project title generated: {title}{Style.RESET_ALL}")
        
        # Generate initial code
        print(f"{Fore.YELLOW}Generating initial code...{Style.RESET_ALL}")
        initial_code = generate_initial_code(task, model)
        print(f"{Fore.GREEN}Initial code generated{Style.RESET_ALL}")
        
        # Iterative improvement process
        for i in range(1, iterations + 1):
            print(f"\n{Fore.CYAN}Starting iteration {i}/{iterations}{Style.RESET_ALL}")
            filename = f"{folder_name}/{title.replace(' ', '')}_{i}.py"
            
            print(f"{Fore.YELLOW}Retrieving previous versions...{Style.RESET_ALL}")
            previous_versions = await get_previous_versions(i)
            print(f"{Fore.GREEN}Previous versions retrieved{Style.RESET_ALL}")
            
            print(f"{Fore.YELLOW}Improving code...{Style.RESET_ALL}")
            improved_code = await improve_code_smart(initial_code, i, previous_versions, model, use_parallel=i > 5)
            print(f"{Fore.GREEN}Code improvement completed{Style.RESET_ALL}")
            
            # Extract, clean, and save the improved code
            extracted_code = extract_and_clean_code(improved_code)
            
            if extracted_code:
                with open(filename, "w") as f:
                    f.write(extracted_code)
                print(f"{Fore.GREEN}Improved code saved to {filename}{Style.RESET_ALL}")
                
                # Test and further improve only on the final iteration, with a maximum of 3 attempts
                if i == iterations:
                    print(f"{Fore.YELLOW}Testing final iteration...{Style.RESET_ALL}")
                    error = test_solution(filename)
                    attempts = 0
                    while error and attempts < 3:
                        print(emoji.emojize(f"{Fore.YELLOW}:warning: Error detected in final iteration. Attempt {attempts + 1}/3{Style.RESET_ALL}"))
                        improved_code = await improve_code_smart(extracted_code, i, previous_versions, model, error, use_parallel=i > 5)
                        extracted_code = extract_and_clean_code(improved_code)
                        if extracted_code:
                            with open(filename, "w") as f:
                                f.write(extracted_code)
                            error = test_solution(filename)
                        else:
                            print(emoji.emojize(f"{Fore.RED}:x: Failed to extract valid code in final iteration, attempt {attempts + 1}{Style.RESET_ALL}"))
                            break
                        attempts += 1
                    
                    if attempts == 3 and error:
                        print(emoji.emojize(f"{Fore.RED}:x: Failed to fix errors after 3 attempts in final iteration{Style.RESET_ALL}"))
                    elif not error:
                        print(f"{Fore.GREEN}Final iteration passed testing{Style.RESET_ALL}")
                
                # Visualize diff if not the first iteration
                if i > 1:
                    print(f"{Fore.YELLOW}Generating diff...{Style.RESET_ALL}")
                    diff = visualize_diff(initial_code, extracted_code)
                    # print(emoji.emojize(f"{Fore.GREEN}:mag: Code changes in iteration {i}:{Style.RESET_ALL}\n{diff}"))
                
                    # Get statistics
                    stats = get_diff_stats(diff)

                    # Get AI summary
                    summary = summarize_changes(diff)

                    print("Diff Statistics:")
                    print(f"Total changes: {stats['total_changes']}")
                    print(f"Additions: {stats['additions']}")
                    print(f"Deletions: {stats['deletions']}")
                    print("\nAI Summary of Changes:")
                    print(summary)
                
                initial_code = extracted_code
            else:
                print(emoji.emojize(f"{Fore.RED}:x: Failed to extract valid code in iteration {i}{Style.RESET_ALL}"))

        print(f"\n{Fore.CYAN}Finalizing project...{Style.RESET_ALL}")

        # Move intermediate iterations to 'versions' folder
        versions_folder = f"{folder_name}/versions"
        os.makedirs(versions_folder, exist_ok=True)
        for i in range(1, iterations):
            old_filename = f"{folder_name}/{title.replace(' ', '')}_{i}.py"
            new_filename = f"{versions_folder}/{title.replace(' ', '')}_{i}.py"
            shutil.move(old_filename, new_filename)
        print(f"{Fore.GREEN}Intermediate versions moved to {versions_folder}{Style.RESET_ALL}")
        
        # Create .gitignore file
        gitignore_content = """
/versions
.DS_Store
"""
        with open(f"{folder_name}/.gitignore", "w") as gitignore_file:
            gitignore_file.write(gitignore_content.strip())
        print(f"{Fore.GREEN}.gitignore file created{Style.RESET_ALL}")
        
        # Generate README.md
        print(f"{Fore.YELLOW}Generating README.md...{Style.RESET_ALL}")
        final_filename = f"{title.replace(' ', '')}_{iterations}.py"
        readme_content = generate_readme(task, title, final_filename)
        with open(f"{folder_name}/README.md", "w") as readme_file:
            readme_file.write(readme_content)
        print(f"{Fore.GREEN}README.md file generated successfully{Style.RESET_ALL}")

        # Generate requirements.txt in the project folder
        print(f"{Fore.YELLOW}Generating requirements.txt...{Style.RESET_ALL}")
        generate_requirements(folder_name)
        print(f"{Fore.GREEN}requirements.txt file generated successfully in {folder_name}{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}Project generation completed successfully!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Your project '{title}' is ready in the '{folder_name}' folder.{Style.RESET_ALL}")

    except ValueError as ve:
        print(emoji.emojize(f"{Fore.RED}:warning: Invalid input: {str(ve)}{Style.RESET_ALL}"))
    except Exception as e:
        print(emoji.emojize(f"{Fore.RED}:boom: An unexpected error occurred: {str(e)}{Style.RESET_ALL}"))
        print(traceback.format_exc())

def extract_and_clean_code(code: str) -> Optional[str]:
    """Extract and clean up the code from the AI's response."""
    try:
        # Remove <answer> tags if present
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_index = code.find(start_tag)
        end_index = code.find(end_tag)
        
        if start_index != -1 and end_index != -1:
            code = code[start_index + len(start_tag):end_index].strip()
        
        # Remove ```python and ``` if present
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Ensure the extracted content is not empty
        if not code:
            logger.warning("Extracted code is empty after cleaning")
            return None
        
        return code
    except Exception as e:
        logger.error(f"Error in extract_and_clean_code: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_iterations(num_iterations=4):
    """
    Run a specified number of iterations, defaulting to 4.
    
    Args:
    num_iterations (int): Number of iterations to run (1-20, default 4)
    
    Returns:
    list: Results of each iteration
    """
    # Ensure num_iterations is within the valid range
    num_iterations = max(1, min(20, num_iterations))
    
    results = []
    for i in range(num_iterations):
        result = f"Iteration {i+1} completed"
        results.append(result)
        print(result)
    
    return results

# Example usage
if __name__ == "__main__":
    # Set up colorful logging
    handler = logging.StreamHandler()
    handler.setFormatter(ColorfulFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.CRITICAL)

    # Run the main function
    asyncio.run(main())